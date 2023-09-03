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

std::string_view read_len_bytes(const char *&start, const char *end) {
  rusty_assert(end - start >= (ssize_t)sizeof(size_t));
  field_size_t len = *(field_size_t *)start;
  start += sizeof(len);
  rusty_assert(end - start >= (ssize_t)len);
  std::string_view ret(start, len);
  start += len;
  return ret;
}

struct BorrowedValue {
  std::vector<std::string_view> fields;
  static BorrowedValue from(const std::vector<std::vector<char>> &fields) {
    std::vector<std::string_view> borrowed;
    for (const std::vector<char> &field : fields)
      borrowed.emplace_back(field.data(), field.size());
    return BorrowedValue{std::move(borrowed)};
  }
  std::vector<char> serialize() {
    auto guard = timers.timer(TimerType::kSerialize).start();
    std::vector<char> ret;
    for (std::string_view field : fields) {
      field_size_t len = field.size();
      ret.insert(ret.end(), (char *)&len, (char *)&len + sizeof(len));
      ret.insert(ret.end(), field.data(), field.data() + len);
    }
    return ret;
  }
  static BorrowedValue deserialize(std::string_view in) {
    auto guard = timers.timer(TimerType::kDeserialize).start();
    std::vector<std::string_view> fields;
    const char *start = in.data();
    const char *end = start + in.size();
    while (start < end) {
      std::string_view field = read_len_bytes(start, end);
      fields.push_back(field);
    }
    rusty_assert(start == end);
    return BorrowedValue{std::move(fields)};
  }
};

enum class OpType {
  INSERT,
  READ,
  UPDATE,
};
struct Insert {
  std::string key;
  std::vector<std::vector<char>> fields;
};
struct Read {
  std::string key;
};
struct Update {
  std::string key;
  std::vector<std::pair<int, std::vector<char>>> fields_to_update;
};
struct Env {
  leveldb::DB *db;
  buffered_channel<std::pair<OpType, uint64_t>> *latency;
};

std::string GenPrismKey(const std::string& key) {
  std::hash<std::string> hasher{};
  char a[8] = {0};
  size_t hv = hasher(key) % 40000000;
  for (int i = 7; i >= 0; --i)
    a[i] = hv >> (7 - i) * 8 & 255;
  return std::string(a, 8);// + key;
}

void do_insert(Env &env, Insert insert) {
  auto guard = timers.timer(TimerType::kInsert).start();
  std::string key = GenPrismKey(insert.key);
  leveldb::Slice key_slice(key);
  std::vector<char> value = BorrowedValue::from(insert.fields).serialize();
  leveldb::Slice value_slice(value.data(), std::min<size_t>(992, value.size()));
  auto put_start = rusty::time::Instant::now();
  auto s = env.db->Put(leveldb::WriteOptions(), key_slice, value_slice);
  auto put_time = put_start.elapsed();
  if (!s.ok()) {
    std::string err = s.ToString();
    rusty_panic("INSERT failed with error: %s\n", err.c_str());
  }
  timers.timer(TimerType::kPut).add(put_time);
  if (env.latency != nullptr) {
    env.latency->push(std::make_pair(OpType::INSERT, put_time.as_nanos()));
  }
}
std::string do_read(Env &env, Read read, std::string& value) {
  auto guard = timers.timer(TimerType::kRead).start();
  std::string key = GenPrismKey(read.key);
  leveldb::Slice key_slice(key);
  auto get_start = rusty::time::Instant::now();
  auto s = env.db->Get(leveldb::ReadOptions(), key_slice, &value);
  auto get_time = get_start.elapsed();
  if (!s.ok()) {
    std::string err = s.ToString();
    rusty_panic("GET failed with error: %s\n", err.c_str());
  }
  timers.timer(TimerType::kGet).add(get_time);
  if (env.latency) {
    env.latency->push(std::make_pair(OpType::READ, get_time.as_nanos()));
  }
  return value;
}
void do_update(Env &env, Update update) {
  auto update_start = rusty::time::Instant::now();
  leveldb::Slice key_slice(update.key);
  std::string value;
  auto get_start = rusty::time::Instant::now();
  auto s = env.db->Get(leveldb::ReadOptions(), key_slice, &value);
  auto get_time = get_start.elapsed();
  if (!s.ok()) {
    std::string err = s.ToString();
    rusty_panic("UPDATE failed when get: %s\n", err.c_str());
  }
  auto values = BorrowedValue::deserialize(value);
  for (const auto &update : update.fields_to_update) {
    int field = update.first;
    std::string_view value(update.second.data(), update.second.size());
    values.fields[field] = value;
  }
  std::vector<char> value_v = values.serialize();
  auto value_slice = leveldb::Slice(value_v.data(), value_v.size());
  auto put_start = rusty::time::Instant::now();
  s = env.db->Put(leveldb::WriteOptions(), key_slice, value_slice);
  auto put_time = put_start.elapsed();
  if (!s.ok()) {
    std::string err = s.ToString();
    rusty_panic("UPDATE failed when put: %s\n", err.c_str());
  }
  auto update_time = update_start.elapsed();
  timers.timer(TimerType::kGet).add(get_time);
  timers.timer(TimerType::kPut).add(put_time);
  timers.timer(TimerType::kUpdate).add(update_time);
  if (env.latency) {
    env.latency->push(std::make_pair(OpType::UPDATE, update_time.as_nanos()));
  }
}

struct WorkEnv {
  FormatType type;
  leveldb::DB *db;
  uint64_t switches;
  const std::filesystem::path &db_path;
  std::atomic<size_t> *progress;
  buffered_channel<std::pair<OpType, uint64_t>> *latency;
};

void print_plain_ans(std::ofstream &out, std::string_view value) {
  BorrowedValue value_parsed = BorrowedValue::deserialize(value);
  rusty_assert(value_parsed.fields.size() == 1);
  out << *value_parsed.fields.begin() << '\n';
}
void print_ycsb_ans(std::ofstream &out, std::string_view value) {
  BorrowedValue value_parsed = BorrowedValue::deserialize(value);
  auto guard = timers.timer(TimerType::kOutput).start();
  out << "[ ";
  const auto &fields = value_parsed.fields;
  for (int i = 0; i < (int)fields.size(); ++i) {
    out.write("field", 5);
    std::string s = std::to_string(i);
    out.write(s.c_str(), s.size());
    out << ' ';
    out.write(fields[i].data(), static_cast<std::streamsize>(fields[i].size()));
    out << ' ';
  }
  out << "]\n";
}
void work(size_t id, WorkEnv work_env,
          buffered_channel<std::variant<Insert, Read, Update>> *in) {
  std::optional<std::ofstream> ans_out =
      work_env.switches & MASK_OUTPUT_ANS
          ? std::optional<std::ofstream>(work_env.db_path /
                                         ("ans_" + std::to_string(id)))
          : std::nullopt;
  Env env{
      .db = work_env.db,
      .latency = work_env.latency,
  };
  std::string prismdb_read_value(4096, 'a');
  for (std::variant<Insert, Read, Update> op : *in) {
    match(
        op, [&](Insert &insert) { do_insert(env, std::move(insert)); },
        [&](Read &read) {
          std::string value = do_read(env, std::move(read), prismdb_read_value);
          if (ans_out.has_value()) {
            switch (work_env.type) {
              case FormatType::Plain:
                print_plain_ans(ans_out.value(), value);
                break;
              case FormatType::YCSB:
                print_ycsb_ans(ans_out.value(), value);
                break;
            }
          }
        },
        [&](Update &update) { do_update(env, std::move(update)); });
    work_env.progress->fetch_add(1, std::memory_order_relaxed);
  }
}

void print_latency(const std::filesystem::path &db_path,
                   buffered_channel<std::pair<OpType, uint64_t>> *latency) {
  if (latency == nullptr) return;
  std::ofstream out(db_path / "latency");
  for (std::pair<OpType, uint64_t> item : *latency) {
    switch (item.first) {
      case OpType::INSERT:
        out << "INSERT";
        break;
      case OpType::READ:
        out << "READ";
        break;
      case OpType::UPDATE:
        out << "UPDATE";
        break;
    }
    out << ' ' << item.second << '\n';
  }
}
void work2(
    FormatType type, leveldb::DB *db, uint64_t switches,
    const std::filesystem::path &db_path, std::atomic<size_t> *progress,
    size_t num_threads,
    void (*parse)(
        std::deque<buffered_channel<std::variant<Insert, Read, Update>>> &)) {
  size_t buf_len = next_power_of_two(num_threads * 10);
  std::optional<buffered_channel<std::pair<OpType, uint64_t>>> latency_chan =
      switches & MASK_LATENCY
          ? std::optional<buffered_channel<std::pair<OpType, uint64_t>>>(
                buf_len)
          : std::nullopt;
  auto latency_chan_ptr =
      latency_chan.has_value() ? &latency_chan.value() : nullptr;
  std::thread latency_printer(print_latency, db_path, latency_chan_ptr);

  WorkEnv env{
      .type = type,
      .db = db,
      .switches = switches,
      .db_path = db_path,
      .progress = progress,
      .latency = latency_chan_ptr,
  };
  std::deque<buffered_channel<std::variant<Insert, Read, Update>>> inputs;
  std::vector<std::thread> workers;
  workers.reserve(num_threads);
  for (size_t i = 0; i < num_threads; ++i) {
    inputs.emplace_back(buf_len);
    workers.emplace_back(work, i, env, &inputs.back());
  }

  parse(inputs);

  for (auto &in : inputs) {
    in.close();
  }
  for (auto &worker : workers) {
    worker.join();
  }
  if (latency_chan_ptr != nullptr) latency_chan_ptr->close();
  latency_printer.join();
}

void parse_plain(
    std::deque<buffered_channel<std::variant<Insert, Read, Update>>> &inputs) {
  size_t num_threads = inputs.size();
  std::hash<std::string> hasher{};
  while (1) {
    std::string op;
    std::cin >> op;
    if (!std::cin) {
      break;
    }
    if (op == "INSERT") {
      std::string key;
      std::cin >> key;
      rusty_assert(std::cin.get() == ' ');
      char c;
      std::vector<char> value;
      while ((c = std::cin.get()) != '\n' && c != EOF) {
        value.push_back(c);
      }
      size_t i = hasher(key) % num_threads;
      inputs[i].push(
          Insert{.key = std::move(key), .fields = {std::move(value)}});
    } else if (op == "READ") {
      std::string key;
      std::cin >> key;
      size_t i = hasher(key) % num_threads;
      inputs[i].push(Read{std::move(key)});
    } else if (op == "UPDATE") {
      rusty_panic("UPDATE in plain format is not supported yet\n");
    } else {
      std::cerr << "Ignore line: " << op;
      std::getline(std::cin, op);  // Skip the rest of the line
      std::cerr << op << std::endl;
    }
  }
}
void work_plain(leveldb::DB *db, uint64_t switches,
                const std::filesystem::path &db_path,
                std::atomic<size_t> *progress, size_t num_threads) {
  work2(FormatType::Plain, db, switches, db_path, progress, num_threads,
        parse_plain);
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
std::vector<std::vector<char>> read_fields_insert(std::istream &in) {
  auto fields = read_fields(in);
  std::sort(fields.begin(), fields.end());
  std::vector<std::vector<char>> ret;
  for (int i = 0; i < (int)fields.size(); ++i) {
    rusty_assert(fields[i].first == i);
    ret.emplace_back(std::move(fields[i].second));
  }
  return ret;
}

std::set<std::string> read_fields_read(std::istream &in) {
  char c;
  do {
    c = static_cast<char>(in.get());
  } while (isspace(c));
  rusty_assert(c == '[', "Invalid KV trace!");
  std::string s;
  std::getline(in, s);
  rusty_assert(s == " <all fields>]",
               "Reading specific fields is not supported yet.");
  return std::set<std::string>();
}

void parse_ycsb(
    std::deque<buffered_channel<std::variant<Insert, Read, Update>>> &inputs) {
  size_t num_threads = inputs.size();
  std::hash<std::string> hasher{};
  while (1) {
    std::string op;
    auto input_op_start = rusty::time::Instant::now();
    std::cin >> op;
    timers.timer(TimerType::kInputOperation).add(input_op_start.elapsed());
    if (!std::cin) {
      break;
    }
    if (op == "INSERT") {
      auto input_start = rusty::time::Instant::now();
      handle_table_name(std::cin);
      std::string key;
      std::cin >> key;
      leveldb::Slice key_slice(key);
      auto field_values = read_fields_insert(std::cin);
      timers.timer(TimerType::kInputInsert).add(input_start.elapsed());

      size_t i = hasher(key) % num_threads;
      inputs[i].push(Insert{std::move(key), std::move(field_values)});
    } else if (op == "READ") {
      auto input_start = rusty::time::Instant::now();
      handle_table_name(std::cin);
      std::string key;
      std::cin >> key;
      leveldb::Slice key_slice(key);
      auto fields = read_fields_read(std::cin);
      timers.timer(TimerType::kInputRead).add(input_start.elapsed());

      size_t i = hasher(key) % num_threads;
      inputs[i].push(Read{std::move(key)});
    } else if (op == "UPDATE") {
      auto input_start = rusty::time::Instant::now();
      handle_table_name(std::cin);
      std::string key;
      std::cin >> key;
      leveldb::Slice key_slice(key);
      auto updates = read_fields(std::cin);
      timers.timer(TimerType::kInputUpdate).add(input_start.elapsed());

      size_t i = hasher(key) % num_threads;
      inputs[i].push(Update{std::move(key), std::move(updates)});
    } else {
      std::cerr << "Ignore line: " << op;
      std::getline(std::cin, op);  // Skip the rest of the line
      std::cerr << op << std::endl;
    }
  }
}
void work_ycsb(leveldb::DB *db, uint64_t switches,
               const std::filesystem::path &db_path,
               std::atomic<size_t> *progress, size_t num_threads) {
  work2(FormatType::YCSB, db, switches, db_path, progress, num_threads,
        parse_ycsb);
}

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

  auto start = std::chrono::steady_clock::now();
  if (format == "plain") {
    work_plain(db, switches, db_path, &progress, num_threads);
  } else if (format == "ycsb") {
    work_ycsb(db, switches, db_path, &progress, num_threads);
  } else {
    rusty_panic("Unrecognized format: %s\n", format.c_str());
  }
  auto end = std::chrono::steady_clock::now();
  std::cerr << (double)std::chrono::duration_cast<std::chrono::nanoseconds>(
                   end - start)
                       .count() /
                   1e9
            << " second(s) for work\n";

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
