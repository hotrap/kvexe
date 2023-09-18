#include <rocksdb/compaction_router.h>
#include <rocksdb/db.h>
#include <rocksdb/filter_policy.h>
#include <rocksdb/options.h>
#include <rocksdb/statistics.h>
#include <rocksdb/table.h>
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
#include <counter_timer_vec.hpp>
#include <cstddef>
#include <cstdlib>
#include <deque>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <memory>
#include <optional>
#include <queue>
#include <random>
#include <set>
#include <string>
#include <thread>
#include <queue>
#include <vector>

#include "viscnts.h"

using boost::fibers::buffered_channel;

typedef uint16_t field_size_t;

enum class FormatType {
  Plain,
  YCSB,
};

std::vector<rocksdb::DbPath> decode_db_paths(std::string db_paths) {
  std::istringstream in(db_paths);
  std::vector<rocksdb::DbPath> ret;
  rusty_assert(in.get() == '{', "Invalid db_paths");
  char c = static_cast<char>(in.get());
  if (c == '}') return ret;
  rusty_assert(c == '{', "Invalid db_paths");
  while (1) {
    std::string path;
    size_t size;
    if (in.peek() == '"') {
      in >> std::quoted(path);
      rusty_assert(in.get() == ',', "Invalid db_paths");
    } else {
      while ((c = static_cast<char>(in.get())) != ',') path.push_back(c);
    }
    in >> size;
    ret.emplace_back(std::move(path), size);
    rusty_assert(in.get() == '}', "Invalid db_paths");
    c = static_cast<char>(in.get());
    if (c != ',') break;
    rusty_assert(in.get() == '{', "Invalid db_paths");
  }
  rusty_assert(c == '}', "Invalid db_paths");
  return ret;
}

int MaxBytesMultiplerAdditional(const rocksdb::Options &options, int level) {
  if (level >= static_cast<int>(
                   options.max_bytes_for_level_multiplier_additional.size())) {
    return 1;
  }
  return options.max_bytes_for_level_multiplier_additional[level];
}

// Return the first level in the last path
int predict_level_assignment(const rocksdb::Options &options) {
  uint32_t p = 0;
  int level = 0;
  assert(!options.db_paths.empty());

  std::cerr << "Predicted level assignment:\n";

  // size remaining in the most recent path
  uint64_t current_path_size = options.db_paths[0].target_size;

  uint64_t level_size;
  int cur_level = 0;

  // max_bytes_for_level_base denotes L1 size.
  // We estimate L0 size to be the same as L1.
  level_size = options.max_bytes_for_level_base;

  // Last path is the fallback
  while (p < options.db_paths.size() - 1) {
    if (current_path_size < level_size) {
      p++;
      current_path_size = options.db_paths[p].target_size;
      continue;
    }
    if (cur_level == level) {
      // Does desired level fit in this path?
      std::cerr << level << ' ' << options.db_paths[p].path << ' ' << level_size
                << std::endl;
      ++level;
    }
    current_path_size -= level_size;
    if (cur_level > 0) {
      if (options.level_compaction_dynamic_level_bytes) {
        // Currently, level_compaction_dynamic_level_bytes is ignored when
        // multiple db paths are specified. https://github.com/facebook/
        // rocksdb/blob/main/db/column_family.cc.
        // Still, adding this check to avoid accidentally using
        // max_bytes_for_level_multiplier_additional
        level_size =
            static_cast<uint64_t>(static_cast<double>(level_size) *
                                  options.max_bytes_for_level_multiplier);
      } else {
        level_size = static_cast<uint64_t>(
            static_cast<double>(level_size) *
            options.max_bytes_for_level_multiplier *
            MaxBytesMultiplerAdditional(options, cur_level));
      }
    }
    cur_level++;
  }
  std::cerr << level << "+ " << options.db_paths[p].path << ' ' << level_size
            << std::endl;
  return level;
}

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
  kInsert = 0,
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
  kRangeHotSize,
  kCountAccessHotPerTier,
  kEnd,
};
constexpr size_t TIMER_NUM = static_cast<size_t>(TimerType::kEnd);
const char *timer_names[] = {
    "Insert",       "Read",
    "Update",       "Put",
    "Get",          "InputOperation",
    "InputInsert",  "InputRead",
    "InputUpdate",  "Output",
    "Serialize",    "Deserialize",
    "RangeHotSize", "CountAccessHotPerTier",
};
static_assert(sizeof(timer_names) == TIMER_NUM * sizeof(const char *));
counter_timer::TypedTimers<TimerType> timers(TIMER_NUM);

static constexpr uint64_t MASK_LATENCY = 0x1;
static constexpr uint64_t MASK_OUTPUT_ANS = 0x2;
static constexpr uint64_t MASK_COUNT_ACCESS_HOT_PER_TIER = 0x4;
static constexpr uint64_t MASK_KEY_HIT_LEVEL = 0x8;

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
    if (!opnum_) return;
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
  bool enable_fast_process{false};
  size_t num_threads{1};
  size_t opblock_size{1024};
};

struct WorkerEnv {
  rocksdb::DB *db;
  rocksdb::ReadOptions read_options;
  rocksdb::WriteOptions write_options;
  bool ignore_notfound{false};
};

void print_ans(std::ofstream &out, std::string value) {
  out << value << '\n';
}

void print_latency(std::ofstream &out, OpType op, uint64_t nanos) {
  switch (op) {
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
  out << ' ' << nanos << '\n';
}

class Tester {
  WorkOptions options_;
  WorkerEnv env_;
  BlockChannel<Operation> channel_;
  std::vector<BlockChannel<Operation>> channel_for_workers_;

  size_t parse_counts_{0};
  std::atomic<size_t> notfound_counts_{0};

 public:
  Tester(const WorkOptions& option) : options_(option), channel_for_workers_(option.num_threads) {
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

  size_t GetOpParseCounts() const {
    return parse_counts_;
  }

  size_t GetNotFoundCounts() const {
    return notfound_counts_.load(std::memory_order_relaxed);
  }

 private:
  void work(size_t id, BlockChannel<Operation>& chan) {
    std::optional<std::ofstream> ans_out =
        options_.switches & MASK_OUTPUT_ANS
            ? std::optional<std::ofstream>(options_.db_path /
                                          ("ans_" + std::to_string(id)))
            : std::nullopt;

    std::optional<std::ofstream> latency_out =
        options_.switches & MASK_LATENCY
            ? std::optional<std::ofstream>(options_.db_path /
                                          ("latency_" + std::to_string(id)))
            : std::nullopt;
    while (true) {
      auto block = chan.GetBlock();
      if (block.empty()) {
        break;
      }
      size_t local_notfound_counts = 0;
      for (Operation& op : block) {
        if (op.type == OpType::INSERT) {
          do_insert(latency_out, op);
        } else if (op.type == OpType::READ) {
          auto value = do_read(latency_out, op);
          if (ans_out) {
            print_ans(ans_out.value(), value);
          }
          if (value == "") {
            local_notfound_counts++;
          }
        } else if (op.type == OpType::UPDATE) {
          do_update(latency_out, op);
        }
        options_.progress->fetch_add(1, std::memory_order_relaxed);
      }
      notfound_counts_ += local_notfound_counts;
    }
  }

  
  void do_insert(std::optional<std::ofstream>& latency, const Operation& insert) {
    auto guard = timers.timer(TimerType::kInsert).start();
    auto put_start = rusty::time::Instant::now();
    auto s = env_.db->Put(env_.write_options, insert.key, rocksdb::Slice(insert.value.data(), insert.value.size()));
    auto put_time = put_start.elapsed();
    if (!s.ok()) {
      std::string err = s.ToString();
      rusty_panic("INSERT failed with error: %s\n", err.c_str());
    }
    timers.timer(TimerType::kPut).add(put_time);
    if (latency) {
      print_latency(latency.value(), OpType::INSERT, put_time.as_nanos());
    }
  }

  std::string do_read(std::optional<std::ofstream>& latency, const Operation& read) {
    auto guard = timers.timer(TimerType::kRead).start();
    std::string value;
    auto get_start = rusty::time::Instant::now();
    auto s = env_.db->Get(env_.read_options, read.key, &value);
    auto get_time = get_start.elapsed();
    if (!s.ok()) {
      if (s.code() == rocksdb::Status::kNotFound && env_.ignore_notfound) {
        value = "";
      } else {
        std::string err = s.ToString();
        rusty_panic("GET failed with error: %s\n", err.c_str());
      }
    }
    timers.timer(TimerType::kGet).add(get_time);
    if (latency) {
      print_latency(latency.value(), OpType::READ, get_time.as_nanos());
    }
    return value;
  }

  void do_update(std::optional<std::ofstream>& latency, const Operation& update) {
    auto guard = timers.timer(TimerType::kUpdate).start();
    auto put_start = rusty::time::Instant::now();
    auto s = env_.db->Put(env_.write_options, update.key, rocksdb::Slice(update.value.data(), update.value.size()));
    auto put_time = put_start.elapsed();
    if (!s.ok()) {
      std::string err = s.ToString();
      rusty_panic("Update failed with error: %s\n", err.c_str());
    }
    timers.timer(TimerType::kUpdate).add(put_time);
    if (latency) {
      print_latency(latency.value(), OpType::UPDATE, put_time.as_nanos());
    }
  }

  void parse() {
    std::vector<BlockChannelClient<Operation>> opblocks;
    if (options_.enable_fast_process) {
      opblocks.emplace_back(&channel_, options_.opblock_size);
    } else {
      for (size_t i = 0; i < options_.num_threads; i++) {
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
        if (options_.format_type == FormatType::YCSB) { 
          handle_table_name(std::cin);
        }
        std::string key;
        std::cin >> key;
        int i = options_.enable_fast_process ? 0 : hasher(key) % options_.num_threads;
        if (options_.format_type == FormatType::Plain) {
          rusty_assert(std::cin.get() == ' ');
          char c;
          std::vector<char> value;
          while ((c = std::cin.get()) != '\n' && c != EOF) {
            value.push_back(c);
          }
          opblocks[i].Push(Operation(OpType::INSERT, std::move(key), std::move(value)));
        } else {
          opblocks[i].Push(Operation(OpType::INSERT, std::move(key), read_value(std::cin)));
        }
        parse_counts_++;
      } else if (op == "READ") {
        std::string key;
        if (options_.format_type == FormatType::YCSB) {        
          handle_table_name(std::cin);
          std::cin >> key;
          read_fields_read(std::cin);
        } else {
          std::cin >> key;
        }
        int i = options_.enable_fast_process ? 0 : hasher(key) % options_.num_threads;
        opblocks[i].Push(Operation(OpType::READ, std::move(key), {}));
        parse_counts_++;

      } else if (op == "UPDATE") {
        if (options_.format_type == FormatType::Plain) {
          rusty_panic("UPDATE in plain format is not supported yet\n");
        }
        handle_table_name(std::cin);
        std::string key;
        std::cin >> key;
        int i = options_.enable_fast_process ? 0 : hasher(key) % options_.num_threads;
        opblocks[i].Push(Operation(OpType::UPDATE, std::move(key), read_value(std::cin)));
        parse_counts_++;
        
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


enum class PerLevelTimerType : size_t {
  kAccess = 0,
  kEnd,
};
constexpr size_t PER_LEVEL_TIMER_NUM =
    static_cast<size_t>(PerLevelTimerType::kEnd);
const char *per_level_timer_names[] = {
    "Access",
};
static_assert(PER_LEVEL_TIMER_NUM ==
              sizeof(per_level_timer_names) / sizeof(const char *));
counter_timer_vec::TypedTimersVector<PerLevelTimerType> per_level_timers(
    PER_LEVEL_TIMER_NUM);

enum class PerTierTimerType : size_t {
  kTransferRange,
  kEnd,
};
constexpr size_t PER_TIER_TIMER_NUM =
    static_cast<size_t>(PerTierTimerType::kEnd);
const char *per_tier_timer_names[] = {
    "TransferRange",
};
static_assert(PER_TIER_TIMER_NUM ==
              sizeof(per_tier_timer_names) / sizeof(const char *));
counter_timer_vec::TypedTimersVector<PerTierTimerType> per_tier_timers(
    PER_TIER_TIMER_NUM);

class RouterVisCnts : public rocksdb::CompactionRouter {
 public:
  RouterVisCnts(
      const rocksdb::Comparator *ucmp, std::filesystem::path dir,
      int tier0_last_level, size_t max_hot_set_size, uint64_t switches,
      buffered_channel<std::pair<std::string, int>> *key_hit_level_chan)
      : switches_(switches),
        vc_(VisCnts::New(ucmp, dir.c_str(), max_hot_set_size)),
        tier0_last_level_(tier0_last_level),
        new_iter_cnt_(0),
        count_access_hot_per_tier_{0, 0},
        key_hit_level_chan_(key_hit_level_chan) {}
  const char *Name() const override { return "RouterVisCnts"; }
  size_t Tier(int level) override {
    if (level <= tier0_last_level_) {
      return 0;
    } else {
      return 1;
    }
  }
  void Access(int level, rocksdb::Slice key, size_t vlen) override {
    size_t tier = Tier(level);

    if (switches_ & MASK_COUNT_ACCESS_HOT_PER_TIER) {
      auto guard = timers.timer(TimerType::kCountAccessHotPerTier).start();
      if (vc_.IsHot(key)) count_access_hot_per_tier_[tier].fetch_add(1);
    }

    auto guard =
        per_level_timers.timer(level, PerLevelTimerType::kAccess).start();
    vc_.Access(key, vlen);
    if (key_hit_level_chan_) {
      key_hit_level_chan_->push(std::make_pair(key.ToString(), level));
    }
  }
  // The returned pointer will stay valid until the next call to Seek or
  // NextHot with this iterator
  rocksdb::CompactionRouter::Iter LowerBound(rocksdb::Slice key) override {
    new_iter_cnt_.fetch_add(1, std::memory_order_relaxed);
    return vc_.LowerBound(key);
  }
  size_t RangeHotSize(rocksdb::Slice smallest,
                      rocksdb::Slice largest) override {
    auto guard = timers.timer(TimerType::kRangeHotSize).start();
    rocksdb::Bound start{
        .user_key = smallest,
        .excluded = false,
    };
    rocksdb::Bound end{
        .user_key = largest,
        .excluded = false,
    };
    rocksdb::RangeBounds range{.start = start, .end = end};
    size_t ret = vc_.RangeHotSize(range);
    return ret;
  }
  size_t new_iter_cnt() {
    return new_iter_cnt_.load(std::memory_order_relaxed);
  }
  std::vector<size_t> hit_count() {
    std::vector<size_t> ret;
    for (size_t i = 0; i < 2; ++i)
      ret.push_back(
          count_access_hot_per_tier_[i].load(std::memory_order_relaxed));
    return ret;
  }

 private:
  const uint64_t switches_;
  VisCnts vc_;
  int tier0_last_level_;

  std::atomic<size_t> new_iter_cnt_;
  std::atomic<size_t> count_access_hot_per_tier_[2];

  buffered_channel<std::pair<std::string, int>> *key_hit_level_chan_;
};

bool has_background_work(rocksdb::DB *db) {
  uint64_t flush_pending;
  uint64_t compaction_pending;
  uint64_t flush_running;
  uint64_t compaction_running;
  bool ok = db->GetIntProperty(
      rocksdb::Slice("rocksdb.mem-table-flush-pending"), &flush_pending);
  rusty_assert(ok, "");
  ok = db->GetIntProperty(rocksdb::Slice("rocksdb.compaction-pending"),
                          &compaction_pending);
  rusty_assert(ok, "");
  ok = db->GetIntProperty(rocksdb::Slice("rocksdb.num-running-flushes"),
                          &flush_running);
  rusty_assert(ok, "");
  ok = db->GetIntProperty(rocksdb::Slice("rocksdb.num-running-compactions"),
                          &compaction_running);
  rusty_assert(ok, "");
  return flush_pending || compaction_pending || flush_running ||
         compaction_running;
}

void wait_for_background_work(rocksdb::DB *db) {
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

template <typename T>
void print_vector(const std::vector<T> &v) {
  std::cerr << "{";
  for (size_t i = 0; i < v.size(); ++i) {
    std::cerr << i << ':' << v[i] << ',';
  }
  std::cerr << "}";
}

auto timestamp_ns() {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
             std::chrono::system_clock::now().time_since_epoch())
      .count();
}
void bg_stat_printer(const rocksdb::Options *options,
                     std::filesystem::path db_path,
                     std::atomic<bool> *should_stop,
                     std::atomic<size_t> *progress) {
  std::ofstream progress_out(db_path / "progress");
  progress_out << "Timestamp(ns) operations-executed\n";
  std::ofstream promoted_2sdlast_out(db_path / "promoted-2sdlast-bytes");
  promoted_2sdlast_out << "Timestamp(ns) num-bytes\n";
  std::ofstream promoted_flush_out(db_path / "promoted-flush-bytes");
  promoted_flush_out << "Timestamp(ns) num-bytes\n";
  while (!should_stop->load(std::memory_order_relaxed)) {
    auto timestamp = timestamp_ns();

    auto value = progress->load(std::memory_order_relaxed);
    progress_out << timestamp << ' ' << value << std::endl;

    auto promoted_2sdlast_bytes =
        options->statistics->getTickerCount(rocksdb::PROMOTED_2SDLAST_BYTES);
    promoted_2sdlast_out << timestamp << ' ' << promoted_2sdlast_bytes
                         << std::endl;

    auto promoted_flush_bytes =
        options->statistics->getTickerCount(rocksdb::PROMOTED_FLUSH_BYTES);
    promoted_flush_out << timestamp << ' ' << promoted_flush_bytes << std::endl;

    std::this_thread::sleep_for(std::chrono::seconds(1));
  }
}

void key_hit_level_print(const std::filesystem::path &dir,
                         buffered_channel<std::pair<std::string, int>> *chan) {
  if (chan == NULL) return;
  std::ofstream out(dir / "key_hit_level");
  for (const auto &p : *chan) {
    out << p.first << ' ' << p.second << std::endl;
  }
}

int main(int argc, char **argv) {
  rocksdb::Options options;

  namespace po = boost::program_options;
  po::options_description desc("Available options");
  std::string format;
  std::string arg_db_path;
  std::string arg_db_paths;
  std::string viscnts_path_str;
  size_t cache_size;
  int compaction_pri;
  double arg_max_hot_set_size;
  std::string arg_switches;
  size_t num_threads;
  size_t max_background_jobs;
  desc.add_options()("help", "Print help message");
  desc.add_options()("cleanup,c", "Empty the directories first.");
  desc.add_options()("enable_fast_process", "Enable fast processing method.");
  desc.add_options()("max_background_jobs", po::value<size_t>(&max_background_jobs)->default_value(1), "max_background_jobs");
  desc.add_options()("format,f",
                     po::value<std::string>(&format)->default_value("ycsb"),
                     "Trace format: plain/ycsb");
  desc.add_options()(
      "use_direct_reads",
      po::value<bool>(&options.use_direct_reads)->default_value(true), "");
  desc.add_options()("db_path",
                     po::value<std::string>(&arg_db_path)->required(),
                     "Path to database");
  desc.add_options()(
      "db_paths", po::value<std::string>(&arg_db_paths)->required(),
      "For example: \"{{/tmp/sd,100000000},{/tmp/cd,1000000000}}\"");
  desc.add_options()("viscnts_path",
                     po::value<std::string>(&viscnts_path_str)->required(),
                     "Path to VisCnts");
  desc.add_options()("cache_size",
                     po::value<size_t>(&cache_size)->default_value(8 << 20),
                     "Capacity of LRU block cache in bytes. Default: 8MiB");
  desc.add_options()("compaction_pri,p",
                     po::value<int>(&compaction_pri)->required(),
                     "Method to pick SST to compact (rocksdb::CompactionPri)");
  desc.add_options()("max_hot_set_size",
                     po::value<double>(&arg_max_hot_set_size)->required(),
                     "Max hot set size in bytes");
  desc.add_options()(
      "switches", po::value<std::string>(&arg_switches)->default_value("none"),
      "Switches for statistics: none/all/<hex value>\n"
      "0x1: Log the latency of each operation\n"
      "0x2: Output the result of READ\n"
      "0x4: count access hot per tier\n"
      "0x8: Log key and the level hit");
  desc.add_options()("num_threads",
                     po::value<size_t>(&num_threads)->default_value(1),
                     "The number of threads to execute the trace\n");
  po::variables_map vm;
  po::store(po::parse_command_line(argc, argv, desc), vm);
  if (vm.count("help")) {
    std::cerr << desc << std::endl;
    return 1;
  }
  po::notify(vm);

  size_t max_hot_set_size = arg_max_hot_set_size;
  uint64_t switches;
  if (arg_switches == "none") {
    switches = 0;
  } else if (arg_switches == "all") {
    switches = 0xf;
  } else {
    std::istringstream in(std::move(arg_switches));
    in >> std::hex >> switches;
  }

  std::filesystem::path db_path(arg_db_path);
  std::filesystem::path viscnts_path(viscnts_path_str);
  options.db_paths = decode_db_paths(arg_db_paths);
  options.compaction_pri = static_cast<rocksdb::CompactionPri>(compaction_pri);
  options.statistics = rocksdb::CreateDBStatistics();
  options.max_background_jobs = max_background_jobs;

  rocksdb::BlockBasedTableOptions table_options;
  table_options.block_cache = rocksdb::NewLRUCache(cache_size);
  table_options.filter_policy.reset(rocksdb::NewBloomFilterPolicy(10, false));
  options.table_factory.reset(
      rocksdb::NewBlockBasedTableFactory(table_options));

  if (vm.count("cleanup")) {
    std::cerr << "Emptying directories\n";
    empty_directory(db_path);
    for (auto path : options.db_paths) {
      empty_directory(path.path);
    }
    empty_directory(viscnts_path_str);
  }

  int first_level_in_cd = predict_level_assignment(options);
  std::ofstream(db_path / "first-level-in-cd")
      << first_level_in_cd << std::endl;

  size_t buf_len = next_power_of_two(num_threads * 10);
  buffered_channel<std::pair<std::string, int>> *key_hit_level_chan;
  if (switches & MASK_KEY_HIT_LEVEL) {
    key_hit_level_chan =
        new buffered_channel<std::pair<std::string, int>>(buf_len);
  } else {
    key_hit_level_chan = nullptr;
  }
  std::thread key_hit_level_printer(key_hit_level_print, db_path,
                                    key_hit_level_chan);

  // options.compaction_router = new RouterTrivial;
  // options.compaction_router = new RouterProb(0.5, 233);
  RouterVisCnts *router = nullptr;
  if (first_level_in_cd != 0) {
    router = new RouterVisCnts(options.comparator, viscnts_path_str,
                               first_level_in_cd - 1, max_hot_set_size,
                               switches, key_hit_level_chan);
    options.compaction_router = router;
  }

  rocksdb::DB *db;
  auto s = rocksdb::DB::Open(options, db_path.string(), &db);
  if (!s.ok()) {
    std::cerr << "Creating database\n";
    options.create_if_missing = true;
    s = rocksdb::DB::Open(options, db_path.string(), &db);
    if (!s.ok()) {
      std::cerr << s.ToString() << std::endl;
      return -1;
    }
  }

  std::atomic<bool> should_stop(false);
  std::atomic<size_t> progress(0);
  std::thread stat_printer(bg_stat_printer, &options, db_path, &should_stop,
                           &progress);

  std::string cmd =
      "pidstat -p " + std::to_string(getpid()) +
      " -Hu 1 | awk '{if(NR>3){print $1,$8; fflush(stdout)}}' > " +
      db_path.c_str() + "/cpu &";
  std::cerr << cmd << std::endl;
  std::system(cmd.c_str());

  // Option for test
  WorkOptions work_option;
  work_option.db = db;
  work_option.switches = switches;
  work_option.db_path = db_path;
  work_option.progress = &progress;
  work_option.num_threads = num_threads;
  work_option.enable_fast_process = vm.count("enable_fast_process");
  work_option.format_type = format == "ycsb" ? FormatType::YCSB : FormatType::Plain;
  Tester tester(work_option);

  auto stats_print_func = [&] (std::ostream& log) {
    log << "Timestamp: " << timestamp_ns() << "\n";
    log << "rocksdb.block.cache.data.miss: "
        << options.statistics->getTickerCount(
              rocksdb::BLOCK_CACHE_DATA_MISS)
        << "\n";
    log << "rocksdb.block.cache.data.hit: "
        << options.statistics->getTickerCount(rocksdb::BLOCK_CACHE_DATA_HIT)
        << "\n";
    log << "rocksdb.bloom.filter.useful: "
        << options.statistics->getTickerCount(rocksdb::BLOOM_FILTER_USEFUL)
        << "\n";
    log << "rocksdb.bloom.filter.full.positive: "
        << options.statistics->getTickerCount(
              rocksdb::BLOOM_FILTER_FULL_POSITIVE)
        << "\n";
    log << "rocksdb.memtable.hit: "
        << options.statistics->getTickerCount(rocksdb::MEMTABLE_HIT)
        << "\n";
    log << "rocksdb.l0.hit: "
        << options.statistics->getTickerCount(rocksdb::GET_HIT_L0)
        << "\n";
    log << "rocksdb.l1.hit: "
        << options.statistics->getTickerCount(rocksdb::GET_HIT_L1)
        << "\n";
    log << "rocksdb.rocksdb.l2andup.hit: "
        << options.statistics->getTickerCount(rocksdb::GET_HIT_L2_AND_UP)
        << "\n";

    /* Statistics of router */
    if (router) {
      log << "New iterator count: " << router->new_iter_cnt() << "\n";
      if (switches & MASK_COUNT_ACCESS_HOT_PER_TIER) {
        auto counters = router->hit_count();
        assert(counters.size() == 2);
        log << "Access hot per tier: " << counters[0] << ' ' << counters[1] << "\n";
      }

      size_t num_tiers = per_tier_timers.len();
      for (size_t tier = 0; tier < num_tiers; ++tier) {
        log << "Tier timers: {tier: " << tier << ", timers: [\n";
        const auto &timers = per_tier_timers.timers(tier);
        size_t num_types = timers.len();
        for (size_t type = 0; type < num_types; ++type) {
          const auto &timer = timers.timer(type);
          log << per_tier_timer_names[type] << ": count " << timer.count()
              << ", total " << timer.time().as_nanos() << "ns,\n";
        }
        log << "]},\n";
      }
      log << "end===\n";

      size_t num_levels = per_level_timers.len();
      for (size_t level = 0; level < num_levels; ++level) {
        log << "Level timers: {level: " << level << ", timers: [\n";
        const auto &timers = per_level_timers.timers(level);
        size_t num_types = timers.len();
        for (size_t type = 0; type < num_types; ++type) {
          const auto &timer = timers.timer(type);
          log << per_level_timer_names[type] << ": count " << timer.count()
              << ", total " << timer.time().as_nanos() << "ns,\n";
        }
        log << "]},\n";
      }
      log << "end===\n";
    }

    /* Timer data */
    std::vector<counter_timer::CountTime> timers_status;
    const auto &ts = timers.timers();
    size_t num_types = ts.len();
    for (size_t i = 0; i < num_types; ++i) {
      const auto &timer = ts.timer(i);
      uint64_t count = timer.count();
      rusty::time::Duration time = timer.time();
      timers_status.push_back(counter_timer::CountTime{count, time});
      log << timer_names[i] << ": count " << count << ", total "
          << time.as_nanos() << "ns\n";
    }

    /* Operation counts*/
    log << "operation counts: " << tester.GetOpParseCounts() << "\n";
    log << "notfound counts: " << tester.GetNotFoundCounts() << "\n";
    log << "stat end===" << std::endl;
  };

  auto period_print_stat = [&] () {
    std::ofstream period_stats(db_path / "period_stats");
    while(!should_stop.load()) {
      stats_print_func(period_stats);
      std::this_thread::sleep_for(std::chrono::seconds(3));
    }
  };

  std::thread period_print_thread(period_print_stat);

  auto start = std::chrono::steady_clock::now();
  tester.Test();
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
  
  stats_print_func(std::cerr);

  std::string rocksdb_stats;
  rusty_assert(db->GetProperty("rocksdb.stats", &rocksdb_stats), "");
  std::ofstream(db_path / "rocksdb-stats.txt") << rocksdb_stats;

  if (key_hit_level_chan) key_hit_level_chan->close();
  key_hit_level_printer.join();
  stat_printer.join();
  period_print_thread.join();
  delete db;
  delete router;

  return 0;
}
