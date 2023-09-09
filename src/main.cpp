#include <leveldb/db.h>
#include <leveldb/filter_policy.h>
#include <leveldb/table.h>
#include <leveldb/cache.h>
#include <leveldb/env.h>
#include <rusty/keyword.h>
#include <rusty/macro.h>
#include <rusty/primitive.h>
#include <rusty/time.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <boost/fiber/buffered_channel.hpp>
#include <boost/program_options/errors.hpp>
#include <boost/program_options/options_description.hpp>
#include <boost/program_options/parsers.hpp>
#include <boost/program_options/variables_map.hpp>
#include <cctype>
#include <chrono>
#include <counter_timer.hpp>
#include <cstddef>
#include <cstdlib>
#include <deque>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <set>
#include <string>
#include <thread>
#include <queue>
#include <vector>

using boost::fibers::buffered_channel;

typedef uint16_t field_size_t;

enum class FormatType {
  Plain,
  YCSB,
};

void empty_directory(std::filesystem::path dir_path) {
  for (auto &path : std::filesystem::directory_iterator(dir_path)) {
    std::filesystem::remove_all(path);
  }
}

bool is_empty_directory(std::string dir_path) {
  auto it = std::filesystem::directory_iterator(dir_path);
  return it == std::filesystem::end(it);
}

enum class TimerType : size_t {
  kInsert,
  kRead,
  kUpdate,
  kPut,
  kGet,
  kInputOperation,
  kInputInsert,
  kInputRead,
  kInputUpdate,
  kOutput,
  kSerialize,
  kDeserialize,
  kEnd,
};
constexpr size_t TIMER_NUM = static_cast<size_t>(TimerType::kEnd);
const char *timer_names[] = {
    "Insert",      "Read",           "Update",      "Put",
    "Get",         "InputOperation", "InputInsert", "InputRead",
    "InputUpdate", "Output",         "Serialize",   "Deserialize",
};
static_assert(sizeof(timer_names) == TIMER_NUM * sizeof(const char *));
counter_timer::TypedTimers<TimerType> timers(TIMER_NUM);

static constexpr uint64_t MASK_LATENCY = 0x1;
static constexpr uint64_t MASK_OUTPUT_ANS = 0x2;

enum class OpType {
  INSERT,
  READ,
  UPDATE,
};

struct Operation {
  OpType type;
  std::string key;
  std::vector<char> value;

  Operation() {}

  Operation(OpType _type, const std::string& _key, const std::vector<char>& _value) :
    type(_type), key(_key), value(_value) {}

  Operation(OpType _type, std::string&& _key, std::vector<char>&& _value) :
    type(_type), key(std::move(_key)), value(std::move(_value)) {}
};

template <typename T>
class BlockChannel {
  std::queue<std::vector<T>> q_;
  std::mutex m_;
  std::condition_variable cv_r_;
  std::condition_variable cv_w_;
  size_t reader_waiting_{0};
  size_t limit_size_{0};
  bool finish_{false};
  bool writer_waiting_{0};
 
 public:
  BlockChannel(size_t limit_size = 4192) : limit_size_(limit_size) {}
  std::vector<T> GetBlock() {
    std::unique_lock lck(m_);
    if (writer_waiting_) {
      if (q_.size() < limit_size_ / 2) {
        cv_w_.notify_one();
      }
    }
    if (q_.size()) {
      auto ret = std::move(q_.front());
      q_.pop();
      return ret;
    }
    if (finish_) {
      return {};
    }
    reader_waiting_ += 1;
    cv_r_.wait(lck, [&]() { return finish_ || !q_.empty(); });
    if (q_.empty() && finish_) {
      return {};
    }
    reader_waiting_ -= 1;
    auto ret = std::move(q_.front());
    q_.pop();
    return ret;
  }

  void PutBlock(std::vector<T>&& block) {
    std::unique_lock lck(m_);
    q_.push(std::move(block));
    if (reader_waiting_) {
      cv_r_.notify_one();
    }
  }

  void PutBlock(const std::vector<T>& block) {
    std::unique_lock lck(m_);
    q_.push(block);
    if (reader_waiting_) {
      cv_r_.notify_one();
    }
    if (q_.size() >= limit_size_) {
      writer_waiting_ = true;
      cv_w_.wait(lck, [&]() { return q_.size() < limit_size_ / 2; });
      writer_waiting_ = false;
    }
  }

  void Finish() {
    finish_ = true;
    cv_r_.notify_all();
  }

};

template<typename T>
class BlockChannelClient {
  BlockChannel<T>* channel_;
  std::vector<T> opblock_;
  size_t opnum_{0};

 public:
  BlockChannelClient(BlockChannel<T>* channel, size_t block_size) : channel_(channel), opblock_(block_size) {}

  void Push(T&& data) {
    opblock_[opnum_++] = std::move(data);
    if (opnum_ == opblock_.size()) {
      channel_->PutBlock(opblock_);
      opnum_ = 0;
    }
  }

  void Flush() {
    channel_->PutBlock(std::vector<T>(opblock_.begin(), opblock_.begin() + opnum_));
    opnum_ = 0;
  }

  void Finish() {
    channel_->Finish();
  }
};

struct WorkOptions {
  FormatType format_type;
  rocksdb::DB *db;
  uint64_t switches;
  std::filesystem::path db_path;
  std::atomic<size_t> *progress;
  BlockChannelClient<std::pair<OpType, uint64_t>> *latency;
  bool enable_fast_process{false};
  size_t num_threads{1};
  size_t opblock_size{1024};
};

struct WorkerEnv {
  rocksdb::DB *db;
  rocksdb::ReadOptions read_options;
  rocksdb::WriteOptions write_options;
  BlockChannelClient<std::pair<OpType, uint64_t>> *latency;
  bool ignore_notfound{false};
};

void print_ans(std::ofstream &out, std::string value) {
  out << value << '\n';
}


class Tester {
  WorkOptions options_;
  WorkerEnv env_;
  BlockChannel<Operation> channel_;
  std::vector<BlockChannel<Operation>> channel_for_workers_;

 public:
  Tester(const WorkOptions& option) : options_(option), channel_for_workers_(option.num_threads) {
    env_.latency = options_.latency;
    env_.db = options_.db;
    env_.read_options = rocksdb::ReadOptions();
    env_.write_options = rocksdb::WriteOptions();
    if (options_.enable_fast_process) {
      env_.ignore_notfound = true;
    }
  }

  void Test() {
    std::vector<std::thread> threads;

    for (size_t i = 0; i < options_.num_threads; i++) {
      threads.emplace_back([this, i]() { work(i, options_.enable_fast_process ? channel_ : channel_for_workers_[i]); });
    }

    parse();

    for (auto& t : threads) t.join();
  }

 private:
  void work(size_t id, BlockChannel<Operation>& chan) {
    std::optional<std::ofstream> ans_out =
        options_.switches & MASK_OUTPUT_ANS
            ? std::optional<std::ofstream>(options_.db_path /
                                          ("ans_" + std::to_string(id)))
            : std::nullopt;
    while (true) {
      auto block = chan.GetBlock();
      if (block.empty()) {
        break;
      }
      for (Operation& op : block) {
        if (op.type == OpType::INSERT) {
          do_insert(env_, op);
        } else if (op.type == OpType::READ) {
          auto value = do_read(env_, op);
          if (ans_out) {
            print_ans(ans_out.value(), value);
          }
        } else if (op.type == OpType::UPDATE) {
          do_update(env_, op);
        }
        options_.progress->fetch_add(1, std::memory_order_relaxed);
      }
    }
  }

  
  void do_insert(WorkerEnv &env, const Operation& insert) {
    auto guard = timers.timer(TimerType::kInsert).start();
    auto put_start = rusty::time::Instant::now();
    auto s = env.db->Put(env.write_options, insert.key, rocksdb::Slice(insert.value.data(), insert.value.size()));
    auto put_time = put_start.elapsed();
    if (!s.ok()) {
      std::string err = s.ToString();
      rusty_panic("INSERT failed with error: %s\n", err.c_str());
    }
    timers.timer(TimerType::kPut).add(put_time);
    if (env.latency != nullptr) {
      env.latency->Push(std::make_pair(OpType::INSERT, put_time.as_nanos()));
    }
  }

  std::string do_read(WorkerEnv &env, const Operation& read) {
    auto guard = timers.timer(TimerType::kRead).start();
    std::string value;
    auto get_start = rusty::time::Instant::now();
    auto s = env.db->Get(env.read_options, read.key, &value);
    auto get_time = get_start.elapsed();
    if (!s.ok()) {
      if (s.code() == rocksdb::Status::kNotFound && env.ignore_notfound) {
        return "NotFound";
      }
      std::string err = s.ToString();
      rusty_panic("GET failed with error: %s\n", err.c_str());
    }
    timers.timer(TimerType::kGet).add(get_time);
    if (env.latency) {
      env.latency->Push(std::make_pair(OpType::READ, get_time.as_nanos()));
    }
    return value;
  }

  void do_update(WorkerEnv &env, const Operation& update) {
    auto guard = timers.timer(TimerType::kUpdate).start();
    auto put_start = rusty::time::Instant::now();
    auto s = env.db->Put(env.write_options, update.key, rocksdb::Slice(update.value.data(), update.value.size()));
    auto put_time = put_start.elapsed();
    if (!s.ok()) {
      std::string err = s.ToString();
      rusty_panic("Update failed with error: %s\n", err.c_str());
    }
    timers.timer(TimerType::kUpdate).add(put_time);
    if (env.latency != nullptr) {
      env.latency->Push(std::make_pair(OpType::UPDATE, put_time.as_nanos()));
    }
  }

  void parse() {
    std::vector<BlockChannelClient<Operation>> opblocks;
    if (options_.enable_fast_process) {
      opblocks.emplace_back(&channel_, options_.opblock_size);
    } else {
      for (int i = 0; i < options_.num_threads; i++) {
        opblocks.emplace_back(&channel_for_workers_[i], options_.opblock_size);
      }
    }

    std::hash<std::string> hasher{};

    while (1) {
      std::string op;
      std::cin >> op;
      if (!std::cin) {
        break;
      }
      if (op == "INSERT") {
        handle_table_name(std::cin);
        std::string key;
        std::cin >> key;
        int i = options_.enable_fast_process ? 0 : hasher(key) % options_.num_threads;
        opblocks[i].Push(Operation(OpType::INSERT, std::move(key), read_value(std::cin)));
      } else if (op == "READ") {
        handle_table_name(std::cin);
        std::string key;
        std::cin >> key;
        read_fields_read(std::cin);
        int i = options_.enable_fast_process ? 0 : hasher(key) % options_.num_threads;
        opblocks[i].Push(Operation(OpType::READ, std::move(key), {}));

      } else if (op == "UPDATE") {
        handle_table_name(std::cin);
        std::string key;
        std::cin >> key;
        int i = options_.enable_fast_process ? 0 : hasher(key) % options_.num_threads;
        opblocks[i].Push(Operation(OpType::UPDATE, std::move(key), read_value(std::cin)));
        
      } else {
        std::cerr << "Ignore line: " << op;
        std::getline(std::cin, op);  // Skip the rest of the line
        std::cerr << op << std::endl;
      }
    }

    for(auto& o : opblocks) {
      o.Flush();
      o.Finish();
    }
  }

  void handle_table_name(std::istream &in) {
    std::string table;
    in >> table;
    rusty_assert(table == "usertable", "Column families not supported yet.");
  }

  std::vector<std::pair<int, std::vector<char>>> read_fields(std::istream &in) {
    std::vector<std::pair<int, std::vector<char>>> ret;
    char c;
    do {
      c = static_cast<char>(in.get());
    } while (isspace(c));
    rusty_assert(c == '[', "Invalid KV trace!");
    rusty_assert(in.get() == ' ', "Invalid KV trace!");
    while (in.peek() != ']') {
      constexpr size_t vallen = 100;
      std::string field;
      std::vector<char> value(vallen);
      while ((c = static_cast<char>(in.get())) != '=') {
        field.push_back(c);
      }
      rusty_assert(field.size() > 5);
      rusty_assert(field.substr(0, 5) == "field");
      int i = std::stoi(field.substr(5));
      rusty_assert(in.read(value.data(), vallen), "Invalid KV trace!");
      rusty_assert(in.get() == ' ', "Invalid KV trace!");
      ret.emplace_back(i, std::move(value));
    }
    in.get();  // ]
    return ret;
  }

  std::vector<char> read_value(std::istream &in) {
    auto fields = read_fields(in);
    std::sort(fields.begin(), fields.end());
    std::vector<char> ret;
    for (int i = 0; i < (int)fields.size(); ++i) {
      rusty_assert(fields[i].first == i);
      ret.insert(ret.end(), fields[i].second.begin(), fields[i].second.end());
    }
    return ret;
  }

  void read_fields_read(std::istream &in) {
    char c;
    do {
      c = static_cast<char>(in.get());
    } while (isspace(c));
    rusty_assert(c == '[', "Invalid KV trace!");
    std::string s;
    std::getline(in, s);
    rusty_assert(s == " <all fields>]",
                "Reading specific fields is not supported yet.");
  }

};


bool has_background_work(leveldb::DB *db) {
  return false;
}

void wait_for_background_work(leveldb::DB *db) {
  while (1) {
    if (has_background_work(db)) {
      std::this_thread::sleep_for(std::chrono::seconds(1));
      continue;
    }
    // The properties are not get atomically. Test for more 20 times more.
    int i;
    for (i = 0; i < 20; ++i) {
      std::this_thread::sleep_for(std::chrono::milliseconds(100));
      if (has_background_work(db)) {
        break;
      }
    }
    if (i == 20) {
      // std::cerr << "There is no background work detected for more than 2
      // seconds. Exiting...\n";
      break;
    }
  }
}

auto timestamp_ns() {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
             std::chrono::system_clock::now().time_since_epoch())
      .count();
}
void bg_stat_printer(std::filesystem::path db_path,
                     std::atomic<bool> *should_stop,
                     std::atomic<size_t> *progress) {
  std::ofstream progress_out(db_path / "progress");
  progress_out << "Timestamp(ns) operations-executed\n";
  while (!should_stop->load(std::memory_order_relaxed)) {
    auto timestamp = timestamp_ns();
    auto value = progress->load(std::memory_order_relaxed);
    progress_out << timestamp << ' ' << value << std::endl;
    std::this_thread::sleep_for(std::chrono::seconds(1));
  }
}

int main(int argc, char **argv) {
  leveldb::Options options;

  namespace po = boost::program_options;
  po::options_description desc("Available options");
  std::string format;
  std::string arg_db_path;
  std::string arg_db_paths;
  size_t cache_size;
  std::string arg_switches;
  size_t num_threads;
  desc.add_options()("help", "Print help message");
  desc.add_options()("cleanup,c", "Empty the directories first.");
  desc.add_options()("enable_fast_process", "Enable fast processing method.");
  desc.add_options()("format,f",
                     po::value<std::string>(&format)->default_value("ycsb"),
                     "Trace format: plain/ycsb");
  desc.add_options()("db_path",
                     po::value<std::string>(&arg_db_path)->required(),
                     "Path to database");
  desc.add_options()(
      "db_paths", po::value<std::string>(&arg_db_paths)->required(),
      "For example: \"{{/tmp/sd,100000000},{/tmp/cd,1000000000}}\"");
  desc.add_options()("cache_size",
                     po::value<size_t>(&cache_size)->default_value(8 << 20),
                     "Capacity of LRU block cache in bytes. Default: 8MiB");
  desc.add_options()(
      "switches", po::value<std::string>(&arg_switches)->default_value("none"),
      "Switches for statistics: none/all/<hex value>\n"
      "0x1: Log the latency of each operation\n"
      "0x2: Output the result of READ");
  desc.add_options()("num_threads",
                     po::value<size_t>(&num_threads)->default_value(1),
                     "The number of threads to execute the trace\n");
  desc.add_options()("migrations_logging",
                      po::value<bool>(&options.migration_logging)->required(), "Option migrations_logging");
  desc.add_options()("read_logging",
                      po::value<bool>(&options.read_logging)->required(), "Option read_logging");
  desc.add_options()("migration_policy",
                      po::value<int>(&options.migration_policy)->required(), "Option migration_policy");
  desc.add_options()("migration_metric",
                      po::value<int>(&options.migration_metric)->required(), "Option migration_metric");
  desc.add_options()("migration_rand_range_num",
                      po::value<int>(&options.migration_rand_range_num)->required(), "Option migration_rand_range_num");
  desc.add_options()("migration_rand_range_size",
                      po::value<int>(&options.migration_rand_range_size)->required(), "Option migration_rand_range_size");
  po::variables_map vm;
  po::store(po::parse_command_line(argc, argv, desc), vm);
  if (vm.count("help")) {
    std::cerr << desc << std::endl;
    return 1;
  }
  po::notify(vm);

  uint64_t switches;
  if (arg_switches == "none") {
    switches = 0;
  } else if (arg_switches == "all") {
    switches = 0x3;
  } else {
    std::istringstream in(std::move(arg_switches));
    in >> std::hex >> switches;
  }

  //PrismDB
  options.env = leveldb::Env::Default();
  options.block_cache = leveldb::NewLRUCache(cache_size);
  options.filter_policy = leveldb::NewBloomFilterPolicy(10);

  std::filesystem::path db_path(arg_db_path);
  // options.db_paths = decode_db_paths(arg_db_paths);
  // options.statistics = leveldb::CreateDBStatistics();

  // leveldb::BlockBasedTableOptions table_options;
  // table_options.block_cache = leveldb::NewLRUCache(cache_size);
  // table_options.filter_policy.reset(leveldb::NewBloomFilterPolicy(10, false));
  // options.table_factory.reset(
  //     leveldb::NewBlockBasedTableFactory(table_options));

  if (vm.count("cleanup")) {
    std::cerr << "Emptying directories\n";
    empty_directory(db_path);
    // for (auto path : options.db_paths) {
    //   empty_directory(path.path);
    // }
    options.create_if_missing = true;
  }

  leveldb::DB *db;
  auto s = leveldb::DB::Open(options, db_path.string(), &db);
  if (!s.ok()) {
    std::cerr << s.ToString() << std::endl;
    return -1;
  }

  std::atomic<bool> should_stop(false);
  std::atomic<size_t> progress(0);
  std::thread stat_printer(bg_stat_printer, db_path, &should_stop, &progress);

  std::string pid = std::to_string(getpid());
  std::string cmd = "pidstat -p " + pid +
                    " -Hu 1 | "
                    "awk '{if(NR>3){print $1,$8; fflush(stdout)}}' > " +
                    db_path.c_str() + "/cpu &";
  std::cerr << cmd << std::endl;
  std::system(cmd.c_str());

  BlockChannel<std::pair<OpType, uint64_t>> latency_chan;
  auto latency_chan_client = std::make_unique<BlockChannelClient<std::pair<OpType, uint64_t>>>(&latency_chan, 65536);

  WorkOptions work_option;
  work_option.db = db;
  work_option.switches = switches;
  work_option.db_path = db_path;
  work_option.progress = &progress;
  work_option.latency = switches & MASK_LATENCY ? latency_chan_client.get() : nullptr;
  work_option.num_threads = num_threads;
  work_option.enable_fast_process = vm.count("enable_fast_process");
  Tester tester(work_option);

  auto start = std::chrono::steady_clock::now();
  tester.Test();
  auto end = std::chrono::steady_clock::now();
  std::cerr << (double)std::chrono::duration_cast<std::chrono::nanoseconds>(
                   end - start)
                       .count() /
                   1e9
            << " second(s) for work\n";
  
  latency_chan_client->Flush();
  latency_chan_client->Finish();

  start = std::chrono::steady_clock::now();
  wait_for_background_work(db);
  end = std::chrono::steady_clock::now();
  std::cerr << (double)std::chrono::duration_cast<std::chrono::nanoseconds>(
                   end - start)
                       .count() /
                   1e9
            << " second(s) waiting for background work\n";

  should_stop.store(true, std::memory_order_relaxed);

  // std::cerr << "rocksdb.block.cache.data.miss: "
  //           << options.statistics->getTickerCount(
  //                  leveldb::BLOCK_CACHE_DATA_MISS)
  //           << std::endl;
  // std::cerr << "rocksdb.block.cache.data.hit: "
  //           << options.statistics->getTickerCount(leveldb::BLOCK_CACHE_DATA_HIT)
  //           << std::endl;
  // std::cerr << "rocksdb.bloom.filter.useful: "
  //           << options.statistics->getTickerCount(leveldb::BLOOM_FILTER_USEFUL)
  //           << std::endl;
  // std::cerr << "rocksdb.bloom.filter.full.positive: "
  //           << options.statistics->getTickerCount(
  //                  leveldb::BLOOM_FILTER_FULL_POSITIVE)
  //           << std::endl;
  // std::cerr << "rocksdb.memtable.hit: "
  //           << options.statistics->getTickerCount(leveldb::MEMTABLE_HIT)
  //           << std::endl;
  // std::cerr << "rocksdb.l0.hit: "
  //           << options.statistics->getTickerCount(leveldb::GET_HIT_L0)
  //           << std::endl;
  // std::cerr << "rocksdb.l1.hit: "
  //           << options.statistics->getTickerCount(leveldb::GET_HIT_L1)
  //           << std::endl;
  // std::cerr << "rocksdb.rocksdb.l2andup.hit: "
  //           << options.statistics->getTickerCount(leveldb::GET_HIT_L2_AND_UP)
  //           << std::endl;

  // std::string rocksdb_stats;
  // rusty_assert(db->GetProperty("rocksdb.stats", &rocksdb_stats), "");
  // std::ofstream(db_path / "rocksdb-stats.txt") << rocksdb_stats;

  std::vector<counter_timer::CountTime> timers_status;
  const auto &ts = timers.timers();
  size_t num_types = ts.len();
  for (size_t i = 0; i < num_types; ++i) {
    const auto &timer = ts.timer(i);
    uint64_t count = timer.count();
    rusty::time::Duration time = timer.time();
    timers_status.push_back(counter_timer::CountTime{count, time});
    std::cerr << timer_names[i] << ": count " << count << ", total "
              << time.as_nanos() << "ns\n";
  }
  std::cerr << "In summary: [\n";
  counter_timer::CountTime input_time =
      timers_status[static_cast<size_t>(TimerType::kInputOperation)] +
      timers_status[static_cast<size_t>(TimerType::kInputInsert)] +
      timers_status[static_cast<size_t>(TimerType::kInputRead)] +
      timers_status[static_cast<size_t>(TimerType::kInputUpdate)];
  std::cerr << "\tInput: count " << input_time.count << ", total "
            << input_time.time.as_nanos() << "ns\n";
  std::cerr << "]\n";

  stat_printer.join();
  delete db;

  return 0;
}
