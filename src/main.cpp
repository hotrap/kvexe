#include <leveldb/cache.h>
#include <leveldb/db.h>
#include <leveldb/env.h>
#include <leveldb/filter_policy.h>
#include <leveldb/table.h>
#include <rusty/keyword.h>
#include <rusty/macro.h>
#include <rusty/sync.h>
#include <rusty/time.h>
#include <sys/resource.h>
#include <unistd.h>
#include <xxhash.h>

#include <algorithm>
#include <atomic>
#include <boost/program_options/errors.hpp>
#include <boost/program_options/options_description.hpp>
#include <boost/program_options/parsers.hpp>
#include <boost/program_options/variables_map.hpp>
#include <cctype>
#include <chrono>
#include <cinttypes>
#include <condition_variable>
#include <counter_timer.hpp>
#include <cstddef>
#include <cstdlib>
#include <deque>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <mutex>
#include <optional>
#include <queue>
#include <set>
#include <string>
#include <thread>
#include <vector>

#include "ycsbgen/ycsbgen.hpp"

static inline auto timestamp_ns() {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
             std::chrono::system_clock::now().time_since_epoch())
      .count();
}

static inline time_t cpu_timestamp_ns(
    clockid_t clockid = CLOCK_THREAD_CPUTIME_ID) {
  struct timespec t;
  if (-1 == clock_gettime(clockid, &t)) {
    perror("clock_gettime");
    rusty_panic();
  }
  return t.tv_sec * 1000000000 + t.tv_nsec;
}

template <typename T>
void print_vector(const std::vector<T>& v) {
  std::cerr << '[';
  for (double x : v) {
    std::cerr << x << ',';
  }
  std::cerr << "]";
}

enum class FormatType {
  Plain,
  PlainLengthOnly,
  YCSB,
};

static inline const char* to_string(YCSBGen::OpType type) {
  switch (type) {
    case YCSBGen::OpType::INSERT:
      return "INSERT";
    case YCSBGen::OpType::READ:
      return "READ";
    case YCSBGen::OpType::UPDATE:
      return "UPDATE";
    case YCSBGen::OpType::RMW:
      return "RMW";
    case YCSBGen::OpType::DELETE:
      return "DELETE";
    case YCSBGen::OpType::SCAN:
      return "SCAN";
  }
  rusty_panic();
}
static inline void print_latency(std::ofstream& out, YCSBGen::OpType op,
                                 uint64_t nanos) {
  out << timestamp_ns() << ' ' << to_string(op) << ' ' << nanos << '\n';
}

enum class TimerType : size_t {
  kPut,
  kGet,
  kDelete,
  kScan,
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
static const char* timer_names[] = {
    "Put",         "Get",       "Delete",      "Scan",   "InputOperation",
    "InputInsert", "InputRead", "InputUpdate", "Output", "Serialize",
    "Deserialize",
};
static_assert(sizeof(timer_names) == TIMER_NUM * sizeof(const char*));
static counter_timer::TypedTimers<TimerType> timers(TIMER_NUM);

static inline void print_timers(std::ostream& out) {
  const auto& ts = timers.timers();
  size_t num_types = ts.size();
  for (size_t i = 0; i < num_types; ++i) {
    const auto& timer = ts[i];
    uint64_t count = timer.count();
    rusty::time::Duration time = timer.time();
    out << timer_names[i] << ": count " << count << ", total "
        << time.as_secs_double() << " s\n";
  }
}

static std::atomic<time_t> put_cpu_nanos(0);
static std::atomic<time_t> get_cpu_nanos(0);
static std::atomic<time_t> delete_cpu_nanos(0);
static std::atomic<time_t> scan_cpu_nanos(0);

constexpr uint64_t MASK_LATENCY = 0x1;
constexpr uint64_t MASK_OUTPUT_ANS = 0x2;

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

  uint64_t queue_empty_when_put_{0};
  uint64_t queue_non_empty_when_put_{0};
  uint64_t reader_blocked_{0};
  uint64_t reader_not_blocked_{0};

 public:
  BlockChannel(size_t limit_size = 64) : limit_size_(limit_size) {}
  std::vector<T> GetBlock() {
    std::unique_lock lck(m_);
    if (writer_waiting_) {
      if (q_.size() < limit_size_ / 2) {
        cv_w_.notify_one();
      }
    }
    if (q_.size()) {
      reader_not_blocked_ += 1;
      auto ret = std::move(q_.front());
      q_.pop();
      return ret;
    }
    if (finish_) {
      return {};
    }
    reader_blocked_ += 1;
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
    if (q_.empty()) {
      queue_empty_when_put_ += 1;
    } else {
      queue_non_empty_when_put_ += 1;
    }
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

  // Must be called when there is no writer.
  uint64_t queue_empty_when_put() const { return queue_empty_when_put_; }
  uint64_t queue_non_empty_when_put() const {
    return queue_non_empty_when_put_;
  }
  uint64_t reader_blocked() const { return reader_blocked_; }
  uint64_t reader_not_blocked() const { return reader_not_blocked_; }
};

template <typename T>
class BlockChannelClient {
  BlockChannel<T>* channel_;
  std::vector<T> opblock_;
  size_t opnum_{0};

 public:
  BlockChannelClient(BlockChannel<T>* channel, size_t block_size)
      : channel_(channel), opblock_(block_size) {}

  void Push(T&& data) {
    opblock_[opnum_++] = std::move(data);
    if (opnum_ == opblock_.size()) {
      channel_->PutBlock(opblock_);
      opnum_ = 0;
    }
  }

  void Flush() {
    channel_->PutBlock(
        std::vector<T>(opblock_.begin(), opblock_.begin() + opnum_));
    opnum_ = 0;
  }

  void Finish() { channel_->Finish(); }
};

struct WorkOptions {
  bool load{false};
  bool run{false};
  std::string load_trace;
  std::string run_trace;
  FormatType format_type;
  leveldb::DB* db;
  uint64_t switches;
  std::filesystem::path db_path;
  bool enable_fast_process{false};
  size_t num_threads{1};
  bool enable_fast_generator{false};
  YCSBGen::YCSBGeneratorOptions ycsb_gen_options;
  bool export_key_only_trace{false};
  bool export_ans_xxh64{false};
  uint64_t sleep_secs_after_load{0};

  size_t num_keys;  // The number of keys after load phase and run phase.
  uint32_t max_value_size;
};

static std::string gen_prism_key(const std::string& key, size_t key_id_pre,
                                 size_t key_num) {
  std::hash<std::string> hasher;
  char a[8] = {0};
  size_t hv = key_id_pre + hasher(key) % key_num;
  for (int i = 7; i >= 0; --i) a[i] = hv >> (7 - i) * 8 & 255;
  return std::string(a, 8);
}

class Tester {
 public:
  Tester(const WorkOptions& option) : options_(option) {
    for (size_t i = 0; i < options_.num_threads; ++i) {
      workers_.emplace_back(*this, i);
    }
  }

  const WorkOptions& work_options() const { return options_; }
  uint64_t progress() const {
    return progress_.load(std::memory_order_relaxed);
  }
  uint64_t progress_get() const {
    return progress_get_.load(std::memory_order_relaxed);
  }

  void Test() {
    std::filesystem::path info_json_path = options_.db_path / "info.json";
    std::ofstream info_json;
    if (options_.load) {
      info_json = std::ofstream(info_json_path);
      info_json << "{" << std::endl;
    } else {
      info_json = std::ofstream(info_json_path, std::ios_base::app);
    }
    rusty::sync::Mutex<std::ofstream> info_json_out(std::move(info_json));

    if (options_.enable_fast_generator) {
      GenerateAndExecute(info_json_out);
    } else {
      ReadAndExecute(info_json_out);
    }

    uint64_t not_found = 0;
    uint64_t scanned = 0;
    for (const auto& worker : workers_) {
      not_found += worker.not_found();
      scanned += worker.scanned();
    }
    if (options_.run) {
      *info_json_out.lock() << "\t\"not-found\": " << not_found << ",\n"
                            << "\t\"scanned-records\": " << scanned << "\n}";
    } else {
      rusty_assert_eq(not_found, (uint64_t)0);
      rusty_assert_eq(scanned, (uint64_t)0);
    }
  }

  void print_other_stats(std::ostream& log) {
    leveldb::DB& db = *options_.db;

    log << "Timestamp: " << timestamp_ns() << "\n";
    std::string leveldb_stats;
    db.GetProperty("leveldb.stats", &leveldb_stats);
    log << "LevelDB stats: " << leveldb_stats << "\n";

    db.ReportMigrationStats(log);

    print_timers(log);

    log << "stat end===" << std::endl;
  }

 private:
  class Worker {
   public:
    Worker(Tester& tester, size_t id)
        : tester_(tester),
          id_(id),
          options_(tester.options_),
          ans_out_(options_.switches & MASK_OUTPUT_ANS
                       ? std::optional<std::ofstream>(
                             options_.db_path / ("ans_" + std::to_string(id)))
                       : std::nullopt) {}

    void load(YCSBGen::YCSBLoadGenerator& loader) {
      while (!loader.IsEOF()) {
        auto op = loader.GetNextOp();
        op.key = gen_prism_key(op.key, 0, options_.num_keys);
        rusty_assert(op.type == YCSBGen::OpType::INSERT);
        do_put(op);
        tester_.progress_.fetch_add(1, std::memory_order_relaxed);
      }
    }
    void prepare_run_phase() {
      if (options_.switches & MASK_LATENCY) {
        latency_out_ = std::make_optional<std::ofstream>(
            options_.db_path / ("latency-" + std::to_string(id_)));
      }
      if (options_.export_ans_xxh64) {
        ans_xxhash_state_ = XXH64_createState();
        rusty_assert(ans_xxhash_state_);
        XXH64_reset(ans_xxhash_state_, 0);
      }
    }
    void finish_run_phase() {
      std::string id = std::to_string(id_);
      if (ans_xxhash_state_) {
        std::ofstream(options_.db_path / ("ans-" + id + ".xxh64"))
            << std::hex << std::setw(16) << std::setfill('0')
            << XXH64_digest(ans_xxhash_state_) << std::endl;
        XXH64_freeState(ans_xxhash_state_);
        ans_xxhash_state_ = nullptr;
      }
    }
    void run(YCSBGen::YCSBRunGenerator& runner) {
      prepare_run_phase();
      std::mt19937_64 rndgen(id_ + options_.ycsb_gen_options.base_seed);

      std::optional<std::ofstream> key_only_trace_out =
          options_.export_key_only_trace
              ? std::make_optional<std::ofstream>(
                    options_.db_path /
                    (std::to_string(id_) + "_key_only_trace"))
              : std::nullopt;
      while (!runner.IsEOF()) {
        auto op = runner.GetNextOp(rndgen);
        op.key = gen_prism_key(op.key, 0, options_.num_keys);
        if (key_only_trace_out.has_value())
          key_only_trace_out.value()
              << to_string(op.type) << ' ' << op.key << '\n';
        std::string value(4096, 0);
        process_op(op, &value);
        tester_.progress_.fetch_add(1, std::memory_order_relaxed);
      }
      finish_run_phase();
    }
    void work(bool run, BlockChannel<YCSBGen::Operation>& chan) {
      if (run) {
        prepare_run_phase();
      }
      std::string value;
      for (;;) {
        auto block = chan.GetBlock();
        if (block.empty()) {
          break;
        }
        for (const YCSBGen::Operation& op : block) {
          std::string value(4096, 0);
          process_op(op, &value);
          tester_.progress_.fetch_add(1, std::memory_order_relaxed);
        }
      }
      if (run) {
        finish_run_phase();
      }
    }

    uint64_t not_found() const { return not_found_; }
    uint64_t scanned() const { return scanned_; }

   private:
    void do_put(const YCSBGen::Operation& put) {
      time_t put_cpu_start = cpu_timestamp_ns();
      auto put_start = rusty::time::Instant::now();
      auto s = options_.db->Put(
          write_options_, put.key,
          leveldb::Slice(
              put.value.data(),
              std::min<size_t>(options_.max_value_size, put.value.size())));
      auto put_time = put_start.elapsed();
      time_t put_cpu_ns = cpu_timestamp_ns() - put_cpu_start;
      if (!s.ok()) {
        std::string err = s.ToString();
        rusty_panic("Put failed with error: %s\n", err.c_str());
      }
      timers.timer(TimerType::kPut).add(put_time);
      put_cpu_nanos.fetch_add(put_cpu_ns, std::memory_order_relaxed);
      if (latency_out_) {
        print_latency(latency_out_.value(), put.type, put_time.as_nanos());
      }
    }

    // Return found or not
    bool do_read(const YCSBGen::Operation& read, std::string* value) {
      time_t get_cpu_start = cpu_timestamp_ns();
      auto get_start = rusty::time::Instant::now();
      auto s = options_.db->Get(read_options_, read.key, value);
      auto get_time = get_start.elapsed();
      time_t get_cpu_ns = cpu_timestamp_ns() - get_cpu_start;
      timers.timer(TimerType::kGet).add(get_time);
      get_cpu_nanos.fetch_add(get_cpu_ns, std::memory_order_relaxed);
      if (latency_out_) {
        print_latency(latency_out_.value(), YCSBGen::OpType::READ,
                      get_time.as_nanos());
      }
      tester_.progress_get_.fetch_add(1, std::memory_order_relaxed);
      if (!s.ok()) {
        if (s.IsNotFound()) {
          return false;
        } else {
          std::string err = s.ToString();
          rusty_panic("GET failed with error: %s\n", err.c_str());
        }
      }
      return true;
    }

    void do_read_modify_write(const YCSBGen::Operation& op) {
      time_t get_cpu_start = cpu_timestamp_ns();
      auto start = rusty::time::Instant::now();
      std::string value;
      auto s = options_.db->Get(read_options_, op.key, &value);
      if (!s.ok()) {
        std::string err = s.ToString();
        rusty_panic("GET failed with error: %s\n", err.c_str());
      }
      time_t put_cpu_start = cpu_timestamp_ns();
      time_t get_cpu_ns = put_cpu_start - get_cpu_start;
      s = options_.db->Put(
          write_options_, op.key,
          leveldb::Slice(
              op.value.data(),
              std::min<size_t>(options_.max_value_size, op.value.size())));
      time_t put_cpu_ns = cpu_timestamp_ns() - put_cpu_start;
      if (!s.ok()) {
        std::string err = s.ToString();
        rusty_panic("Update failed with error: %s\n", err.c_str());
      }
      get_cpu_nanos.fetch_add(get_cpu_ns, std::memory_order_relaxed);
      put_cpu_nanos.fetch_add(put_cpu_ns, std::memory_order_relaxed);
      if (latency_out_) {
        print_latency(latency_out_.value(), YCSBGen::OpType::RMW,
                      start.elapsed().as_nanos());
      }
      tester_.progress_get_.fetch_add(1, std::memory_order_relaxed);
    }

    void do_delete(const YCSBGen::Operation& op) {
      time_t cpu_start = cpu_timestamp_ns();
      auto start = rusty::time::Instant::now();
      auto s = options_.db->Delete(write_options_, op.key);
      auto time = start.elapsed();
      time_t cpu_ns = cpu_timestamp_ns() - cpu_start;
      if (!s.ok()) {
        std::string err = s.ToString();
        rusty_panic("Delete failed with error: %s\n", err.c_str());
      }
      timers.timer(TimerType::kDelete).add(time);
      delete_cpu_nanos.fetch_add(cpu_ns, std::memory_order_relaxed);
      if (latency_out_) {
        print_latency(latency_out_.value(), YCSBGen::OpType::DELETE,
                      time.as_nanos());
      }
    }

    void do_scan(const YCSBGen::Operation& op) {
      time_t cpu_start = cpu_timestamp_ns();
      auto start = rusty::time::Instant::now();
      {
        std::unique_ptr<leveldb::Iterator> it(
            options_.db->NewIterator(read_options_));
        it->Seek(op.key);
        for (size_t i = 0; i < op.scan_len && it->Valid(); ++i) {
          ++scanned_;
          it->Next();
        }
      }
      auto time = start.elapsed();
      time_t cpu_ns = cpu_timestamp_ns() - cpu_start;
      timers.timer(TimerType::kScan).add(time);
      scan_cpu_nanos.fetch_add(cpu_ns, std::memory_order_relaxed);
      if (latency_out_) {
        print_latency(latency_out_.value(), YCSBGen::OpType::SCAN,
                      time.as_nanos());
      }
    }

    void process_op(const YCSBGen::Operation& op, std::string* value) {
      switch (op.type) {
        case YCSBGen::OpType::INSERT:
        case YCSBGen::OpType::UPDATE:
          do_put(op);
          break;
        case YCSBGen::OpType::READ: {
          bool found = do_read(op, value);
          std::string_view ans;
          if (found) {
            ans = std::string_view(value->data(), value->size());
          } else {
            ans = std::string_view(nullptr, 0);
          };
          if (ans_xxhash_state_) {
            rusty_assert_eq(
                XXH64_update(ans_xxhash_state_, ans.data(), ans.size()),
                XXH_OK);
            static const char delimiter = '\n';
            rusty_assert_eq(
                XXH64_update(ans_xxhash_state_, &delimiter, sizeof(delimiter)),
                XXH_OK);
          }
          if (ans_out_) {
            ans_out_.value() << ans << '\n';
          }
          if (!found) {
            not_found_ += 1;
          }
        } break;
        case YCSBGen::OpType::RMW:
          do_read_modify_write(op);
          break;
        case YCSBGen::OpType::DELETE:
          do_delete(op);
          break;
        case YCSBGen::OpType::SCAN:
          do_scan(op);
          break;
      }
    }

    Tester& tester_;
    size_t id_;
    const WorkOptions& options_;
    leveldb::ReadOptions read_options_;
    leveldb::WriteOptions write_options_;

    uint64_t not_found_{0};
    uint64_t scanned_{0};
    XXH64_state_t* ans_xxhash_state_{nullptr};
    std::optional<std::ofstream> ans_out_;
    std::optional<std::ofstream> latency_out_;
  };

  void parse(bool run, std::istream& trace) {
    size_t num_channels =
        options_.enable_fast_process ? 1 : options_.num_threads;
    std::vector<BlockChannel<YCSBGen::Operation>> channel_for_workers(
        num_channels);

    std::vector<BlockChannelClient<YCSBGen::Operation>> opblocks;
    for (auto& channel : channel_for_workers) {
      opblocks.emplace_back(&channel, 1024);
    }

    std::vector<std::thread> threads;
    for (size_t i = 0; i < options_.num_threads; i++) {
      threads.emplace_back([this, run, &channel_for_workers, i]() {
        size_t index = options_.enable_fast_process ? 0 : i;
        workers_[i].work(run, channel_for_workers[index]);
      });
    }

    std::hash<std::string> hasher{};

    size_t key_num = options_.num_keys;
    size_t key_id_pre = 0;

    uint64_t parse_counts = 0;
    while (1) {
      std::string op;
      trace >> op;
      if (!trace) {
        break;
      }
      if (op == "INSERT" || op == "UPDATE") {
        YCSBGen::OpType type;
        if (op == "INSERT") {
          type = YCSBGen::OpType::INSERT;
        } else {
          type = YCSBGen::OpType::UPDATE;
        }
        if (options_.format_type == FormatType::YCSB) {
          handle_table_name(trace);
        }
        std::string key;
        trace >> key;
        int i = options_.enable_fast_process
                    ? 0
                    : hasher(key) % options_.num_threads;
        std::vector<char> value;
        if (options_.format_type == FormatType::YCSB) {
          value = read_value(trace);
        } else {
          rusty_assert_eq(trace.get(), ' ');
          char c;
          if (options_.format_type == FormatType::Plain) {
            while ((c = trace.get()) != '\n' && c != EOF) {
              value.push_back(c);
            }
          } else {
            size_t value_length;
            trace >> value_length;
            value.resize(value_length);
            int ret = snprintf(value.data(), value.size(), "%s%" PRIu64,
                               run ? "load-" : "run-", parse_counts + 1);
            rusty_assert(ret > 0);
            if ((size_t)ret < value_length) {
              memset(value.data() + ret, '-', value_length - ret);
            }
          }
        }
        opblocks[i].Push(YCSBGen::Operation(
            type, gen_prism_key(key, key_id_pre, key_num), std::move(value)));
      } else if (op == "READ") {
        std::string key;
        if (options_.format_type == FormatType::YCSB) {
          handle_table_name(trace);
          trace >> key;
          read_fields_read(trace);
        } else {
          trace >> key;
        }
        int i = options_.enable_fast_process
                    ? 0
                    : hasher(key) % options_.num_threads;
        key = gen_prism_key(key, key_id_pre, key_num);
        opblocks[i].Push(
            YCSBGen::Operation(YCSBGen::OpType::READ, std::move(key), {}));
      } else if (op == "DELETE") {
        std::string key;
        rusty_assert(options_.format_type == FormatType::Plain ||
                     options_.format_type == FormatType::PlainLengthOnly);
        trace >> key;
        int i = options_.enable_fast_process
                    ? 0
                    : hasher(key) % options_.num_threads;
        opblocks[i].Push(
            YCSBGen::Operation(YCSBGen::OpType::DELETE,
                               gen_prism_key(key, key_id_pre, key_num), {}));
      } else {
        std::cerr << "Ignore line: " << op;
        std::getline(trace, op);  // Skip the rest of the line
        std::cerr << op << std::endl;
        continue;
      }
      parse_counts += 1;
    }

    for (auto& o : opblocks) {
      o.Flush();
      o.Finish();
    }

    for (auto& t : threads) t.join();

    uint64_t queue_empty_when_put = 0;
    uint64_t queue_non_empty_when_put = 0;
    uint64_t reader_blocked = 0;
    uint64_t reader_not_blocked = 0;
    for (const auto& channel : channel_for_workers) {
      queue_empty_when_put += channel.queue_empty_when_put();
      queue_non_empty_when_put += channel.queue_non_empty_when_put();
      reader_blocked += channel.reader_blocked();
      reader_not_blocked += channel.reader_not_blocked();
    }
    std::cerr << "Queue empty when put: " << queue_empty_when_put << std::endl;
    std::cerr << "Queue non-empty when put: " << queue_non_empty_when_put
              << std::endl;
    std::cerr << "Reader blocked: " << reader_blocked << std::endl;
    std::cerr << "Reader not blocked: " << reader_not_blocked << std::endl;
  }

  void handle_table_name(std::istream& in) {
    std::string table;
    in >> table;
    rusty_assert(table == "usertable", "Column families not supported yet.");
  }

  std::vector<std::pair<int, std::vector<char>>> read_fields(std::istream& in) {
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

  std::vector<char> read_value(std::istream& in) {
    auto fields = read_fields(in);
    std::sort(fields.begin(), fields.end());
    std::vector<char> ret;
    for (int i = 0; i < (int)fields.size(); ++i) {
      rusty_assert(fields[i].first == i);
      ret.insert(ret.end(), fields[i].second.begin(), fields[i].second.end());
    }
    return ret;
  }

  void read_fields_read(std::istream& in) {
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

  void finish_load_phase(const rusty::sync::Mutex<std::ofstream>& info_json_out,
                         rusty::time::Instant load_start) {
    *info_json_out.lock() << "\t\"load-time(secs)\": "
                          << load_start.elapsed().as_secs_double() << ','
                          << std::endl;

    std::ofstream other_stats_out(options_.db_path /
                                  "other-stats-load-finish.txt");
    print_other_stats(other_stats_out);

    if (options_.sleep_secs_after_load) {
      std::this_thread::sleep_for(
          std::chrono::seconds(options_.sleep_secs_after_load));
    }
  }

  void prepare_run_phase(
      const rusty::sync::Mutex<std::ofstream>& info_json_out) {
    options_.db->SetDbMode(false);
    options_.db->ResetMigrationStats();

    const auto& ts = timers.timers();
    for (const auto& timer : ts) {
      timer.reset();
    }

    *info_json_out.lock() << "\t\"run-start-timestamp(ns)\": " << timestamp_ns()
                          << ',' << std::endl;
  }

  void finish_run_phase(const rusty::sync::Mutex<std::ofstream>& info_json_out,
                        rusty::time::Instant run_start) {
    *info_json_out.lock() << "\t\"run-end-timestamp(ns)\": " << timestamp_ns()
                          << ",\n"
                          << "\t\"run-time(secs)\": "
                          << run_start.elapsed().as_secs_double() << ','
                          << std::endl;
  }

  void GenerateAndExecute(
      const rusty::sync::Mutex<std::ofstream>& info_json_out) {
    std::vector<std::thread> threads;

    std::cerr << "YCSB Options: " << options_.ycsb_gen_options.ToString()
              << std::endl;
    uint64_t now_key_num =
        options_.load ? 0 : options_.ycsb_gen_options.record_count;
    YCSBGen::YCSBLoadGenerator loader(options_.ycsb_gen_options, now_key_num);
    if (options_.load) {
      *info_json_out.lock()
          << "\t\"num-load-op\": " << options_.ycsb_gen_options.record_count
          << ',' << std::endl;

      auto load_start = rusty::time::Instant::now();
      for (size_t i = 0; i < options_.num_threads; ++i) {
        threads.emplace_back(
            [this, &loader, i]() { workers_[i].load(loader); });
      }
      for (auto& t : threads) t.join();
      threads.clear();
      finish_load_phase(info_json_out, load_start);
    }

    if (options_.run) {
      prepare_run_phase(info_json_out);
      auto run_start = rusty::time::Instant::now();
      YCSBGen::YCSBRunGenerator runner = loader.into_run_generator();
      std::vector<clockid_t> clockids;
      std::atomic<size_t> finished(0);
      bool permit_join = false;
      std::condition_variable cv;
      std::mutex mu;
      for (size_t i = 0; i < options_.num_threads; ++i) {
        threads.emplace_back(
            [this, &runner, i, &finished, &permit_join, &cv, &mu]() {
              workers_[i].run(runner);
              finished.fetch_add(1, std::memory_order_relaxed);
              std::unique_lock lock(mu);
              cv.wait(lock, [&permit_join]() { return permit_join; });
            });
        pthread_t thread_id = threads[i].native_handle();
        clockid_t clock_id;
        int ret = pthread_getcpuclockid(thread_id, &clock_id);
        if (ret) {
          switch (ret) {
            case ENOENT:
              rusty_panic(
                  "pthread_getcpuclockid: Per-thread CPU time clocks are not "
                  "supported by the system.");
            case ESRCH:
              rusty_panic(
                  "pthread_getcpuclockid: No thread with the ID %lu could "
                  "be found.",
                  thread_id);
            default:
              rusty_panic("pthread_getcpuclockid returns %d", ret);
          }
        }
        clockids.push_back(clock_id);
      }
      std::ofstream out(options_.db_path / "worker-cpu-nanos");
      out << "Timestamp(ns) cpu-time(ns)\n";
      auto interval = rusty::time::Duration::from_secs(1);
      auto next_begin = rusty::time::Instant::now() + interval;
      std::vector<uint64_t> ori_cpu_timestamp_ns;
      for (size_t i = 0; i < clockids.size(); ++i) {
        ori_cpu_timestamp_ns.push_back(cpu_timestamp_ns(clockids[i]));
      }
      while (finished.load(std::memory_order_relaxed) != threads.size()) {
        auto sleep_time =
            next_begin.checked_duration_since(rusty::time::Instant::now());
        if (sleep_time.has_value()) {
          std::this_thread::sleep_for(
              std::chrono::nanoseconds(sleep_time.value().as_nanos()));
        }
        next_begin += interval;

        auto timestamp = timestamp_ns();

        uint64_t nanos = 0;
        for (size_t i = 0; i < clockids.size(); ++i) {
          nanos += cpu_timestamp_ns(clockids[i]) - ori_cpu_timestamp_ns[i];
        }
        out << timestamp << ' ' << nanos << std::endl;
      }
      {
        std::unique_lock<std::mutex> lock(mu);
        permit_join = true;
      }
      cv.notify_all();
      for (auto& t : threads) t.join();
      finish_run_phase(info_json_out, run_start);
    }
  }

  void ReadAndExecute(const rusty::sync::Mutex<std::ofstream>& info_json_out) {
    if (options_.load) {
      std::optional<std::ifstream> trace_file;
      if (!options_.load_trace.empty()) {
        trace_file = std::ifstream(options_.load_trace);
        rusty_assert(trace_file.value());
      }
      std::istream& trace =
          trace_file.has_value() ? trace_file.value() : std::cin;

      auto start = rusty::time::Instant::now();
      parse(false, trace);
      finish_load_phase(info_json_out, start);
    }
    if (options_.run) {
      std::optional<std::ifstream> trace_file;
      if (!options_.run_trace.empty()) {
        trace_file = std::ifstream(options_.run_trace);
        rusty_assert(trace_file.value());
      }
      std::istream& trace =
          trace_file.has_value() ? trace_file.value() : std::cin;

      prepare_run_phase(info_json_out);
      auto start = rusty::time::Instant::now();
      parse(true, trace);
      finish_run_phase(info_json_out, start);
    }
  }

  WorkOptions options_;
  std::vector<Worker> workers_;

  std::atomic<uint64_t> progress_{0};
  std::atomic<uint64_t> progress_get_{0};
};

static inline void empty_directory(std::filesystem::path dir_path) {
  for (auto &path : std::filesystem::directory_iterator(dir_path)) {
    std::filesystem::remove_all(path);
  }
}

std::vector<std::pair<std::filesystem::path, size_t>> decode_db_paths(
    std::string db_paths) {
  std::istringstream in(db_paths);
  std::vector<std::pair<std::filesystem::path, size_t>> ret;
  rusty_assert_eq(in.get(), '{', "Invalid db_paths");
  char c = static_cast<char>(in.get());
  if (c == '}') return ret;
  rusty_assert_eq(c, '{', "Invalid db_paths");
  while (1) {
    std::string path;
    uint64_t size;
    if (in.peek() == '"') {
      in >> std::quoted(path);
      rusty_assert_eq(in.get(), ',', "Invalid db_paths");
    } else {
      while ((c = static_cast<char>(in.get())) != ',') path.push_back(c);
    }
    in >> size;
    ret.emplace_back(std::move(path), size);
    rusty_assert_eq(in.get(), '}', "Invalid db_paths");
    c = static_cast<char>(in.get());
    if (c != ',') break;
    rusty_assert_eq(in.get(), '{', "Invalid db_paths");
  }
  rusty_assert_eq(c, '}', "Invalid db_paths");
  return ret;
}

void bg_stat_printer(Tester *tester, std::atomic<bool> *should_stop) {
  const WorkOptions &work_options = tester->work_options();
  const std::filesystem::path &db_path = work_options.db_path;

  char buf[16];

  std::string pid = std::to_string(getpid());

  std::ofstream progress_out(db_path / "progress");
  progress_out << "Timestamp(ns) operations-executed get\n";

  std::ofstream mem_out(db_path / "mem");
  std::string mem_command = "ps -q " + pid + " -o rss | tail -n 1";
  mem_out << "Timestamp(ns) RSS(KiB) max-rss(KiB)\n";
  struct rusage rusage;

  auto cputimes_path = db_path / "cputimes";
  std::string cputimes_command = "echo $(ps -q " + pid +
                                 " -o cputimes | tail -n 1) >> " +
                                 cputimes_path.c_str();
  std::ofstream(cputimes_path) << "Timestamp(ns) cputime(s)\n";

  std::ofstream timers_out(db_path / "timers");
  timers_out << "Timestamp(ns) put-cpu-nanos "
                "get-cpu-nanos delete-cpu-nanos\n";

  auto interval = rusty::time::Duration::from_secs(1);
  auto next_begin = rusty::time::Instant::now() + interval;
  while (!should_stop->load(std::memory_order_relaxed)) {
    auto timestamp = timestamp_ns();
    progress_out << timestamp << ' ' << tester->progress() << ' '
                 << tester->progress_get() << std::endl;

    FILE *pipe = popen(mem_command.c_str(), "r");
    if (pipe == NULL) {
      perror("popen");
      rusty_panic();
    }
    rusty_assert(fgets(buf, sizeof(buf), pipe) != NULL, "buf too short");
    size_t buflen = strlen(buf);
    rusty_assert(buflen > 0);
    rusty_assert(buf[--buflen] == '\n');
    buf[buflen] = 0;
    if (-1 == pclose(pipe)) {
      perror("pclose");
      rusty_panic();
    }
    if (-1 == getrusage(RUSAGE_SELF, &rusage)) {
      perror("getrusage");
      rusty_panic();
    }
    mem_out << timestamp << ' ' << buf << ' ' << rusage.ru_maxrss << std::endl;

    std::ofstream(cputimes_path, std::ios_base::app) << timestamp << ' ';
    std::system(cputimes_command.c_str());

    timers_out << timestamp << ' '
               << put_cpu_nanos.load(std::memory_order_relaxed) << ' '
               << get_cpu_nanos.load(std::memory_order_relaxed) << ' '
               << delete_cpu_nanos.load(std::memory_order_relaxed) << std::endl;

    auto sleep_time =
        next_begin.checked_duration_since(rusty::time::Instant::now());
    if (sleep_time.has_value()) {
      std::this_thread::sleep_for(
          std::chrono::nanoseconds(sleep_time.value().as_nanos()));
    }
    next_begin += interval;
  }
}

int main(int argc, char **argv) {
  std::ios::sync_with_stdio(false);
  std::cin.tie(0);
  std::cout.tie(0);

  leveldb::Options options;
  WorkOptions work_options;

  namespace po = boost::program_options;
  po::options_description desc("Available options");
  std::string format;
  std::string arg_switches;

  std::string arg_db_path;
  std::string arg_db_paths;
  size_t cache_size;

  size_t num_keys;

  // Options of executor
  desc.add_options()("help", "Print help message");
  desc.add_options()("format,f",
                     po::value<std::string>(&format)->default_value("ycsb"),
                     "Trace format: plain/plain-length-only/ycsb");
  desc.add_options()(
      "load", po::value<std::string>()->implicit_value(""),
      "Execute the load phase. If a trace is provided with this option, "
      "execute the trace in the load phase. Will empty the directories first.");
  desc.add_options()(
      "run", po::value<std::string>()->implicit_value(""),
      "Execute the run phase. If a trace is provided with this option, execute "
      "the trace in the run phase. "
      "If --load is not provided, the run phase will be executed directly "
      "without executing the load phase, and the directories won't be cleaned "
      "up. "
      "If none of --load and --run is provided, the both phases will be "
      "executed.");
  desc.add_options()("switches",
                     po::value(&arg_switches)->default_value("none"),
                     "Switches for statistics: none/all/<hex value>\n"
                     "0x1: Log the latency of each operation\n"
                     "0x2: Output the result of READ");
  desc.add_options()("num_threads",
                     po::value(&work_options.num_threads)->default_value(1),
                     "The number of threads to execute the trace\n");
  desc.add_options()("enable_fast_process",
                     "Enable fast process including ignoring kNotFound and "
                     "pushing operations in one channel.");
  desc.add_options()("enable_fast_generator", "Enable fast generator");
  desc.add_options()("workload_file", po::value<std::string>(),
                     "Workload file used in built-in generator");
  desc.add_options()("export_key_only_trace",
                     "Export key-only trace generated by built-in generator.");
  desc.add_options()("export_ans_xxh64", "Export xxhash of ans");
  desc.add_options()("sleep_secs_after_load",
                     po::value<uint64_t>(&work_options.sleep_secs_after_load),
                     "Number of seconds to sleep after the load phase.");

  // Options of rocksdb
  desc.add_options()("db_path",
                     po::value<std::string>(&arg_db_path)->required(),
                     "Path to database");
  desc.add_options()(
      "db_paths", po::value<std::string>(&arg_db_paths)->required(),
      "For example: \"{{/tmp/sd,100000000},{/tmp/cd,1000000000}}\"");
  desc.add_options()("cache_size",
                     po::value<size_t>(&cache_size)->default_value(8 << 20),
                     "Capacity of LRU block cache in bytes. Default: 8MiB");

  // Options of PrismDB
  desc.add_options()("num_keys", po::value<size_t>(&num_keys)->required(),
                     "The number of keys.\n");
  desc.add_options()("migrations_logging",
                     po::value<bool>(&options.migration_logging)->required(),
                     "Option migrations_logging");
  desc.add_options()("read_logging",
                     po::value<bool>(&options.read_logging)->required(),
                     "Option read_logging");
  desc.add_options()("migration_policy",
                     po::value<int>(&options.migration_policy)->required(),
                     "Option migration_policy");
  desc.add_options()("migration_metric",
                     po::value<int>(&options.migration_metric)->required(),
                     "Option migration_metric");
  desc.add_options()(
      "migration_rand_range_num",
      po::value<int>(&options.migration_rand_range_num)->required(),
      "Option migration_rand_range_num");
  desc.add_options()(
      "migration_rand_range_size",
      po::value<int>(&options.migration_rand_range_size)->required(),
      "Option migration_rand_range_size");
  desc.add_options()(
      "optane_threshold",
      po::value<float>(&options.optaneThreshold)->default_value(0.15),
      "Optane threshold.");
  desc.add_options()("slab_dir",
                     po::value<std::string>(&options.slab_dir)->required(),
                     "Directory of slabs.");
  desc.add_options()("pop_cache_size",
                     po::value<uint32_t>(&options.popCacheSize)->required(),
                     "size of popularity cache.");
  desc.add_options()(
      "read_dominated_threshold",
      po::value(&options.read_dominated_threshold)->default_value(0.95),
      "read_dominated_threshold");
  desc.add_options()(
      "stop_upsert_trigger",
      po::value(&options.stop_upsert_trigger)->default_value(250 * 1e6),
      "stop_upsert_trigger");
  desc.add_options()("max_kvsize_bytes", po::value(&options.maxKVSizeBytes),
                     "maxKVSizeBytes");

  po::variables_map vm;
  po::store(po::parse_command_line(argc, argv, desc), vm);
  if (vm.count("help")) {
    std::cerr << desc << std::endl;
    return 1;
  }
  po::notify(vm);

  if (vm.count("load")) {
    work_options.load = true;
    work_options.load_trace = vm["load"].as<std::string>();
  }
  if (vm.count("run")) {
    work_options.run = true;
    work_options.run_trace = vm["run"].as<std::string>();
  }
  if (work_options.load == false && work_options.run == false) {
    work_options.load = true;
    work_options.run = true;
  }

  uint64_t switches;
  if (arg_switches == "none") {
    switches = 0;
  } else if (arg_switches == "all") {
    switches = 0x3;
  } else {
    std::istringstream in(std::move(arg_switches));
    in >> std::hex >> switches;
  }

  work_options.enable_fast_generator = vm.count("enable_fast_generator");
  if (work_options.enable_fast_generator) {
    std::string workload_file = vm["workload_file"].as<std::string>();
    work_options.ycsb_gen_options =
        YCSBGen::YCSBGeneratorOptions::ReadFromFile(workload_file);
    work_options.export_key_only_trace = vm.count("export_key_only_trace");

    // PrismDB
    // 1.1 to reduce hash collision.
    work_options.num_keys =
        (work_options.ycsb_gen_options.record_count +
         work_options.ycsb_gen_options.operation_count *
             work_options.ycsb_gen_options.insert_proportion) *
        1.1;
    num_keys = work_options.num_keys;
    options.stop_upsert_trigger =
        0.7 * work_options.ycsb_gen_options.operation_count;
  } else {
    rusty_assert(vm.count("workload_file") == 0,
                 "workload_file only works with built-in generator!");
    rusty_assert(vm.count("export_key_only_trace") == 0,
                 "export_key_only_trace only works with built-in generator!");
    work_options.ycsb_gen_options = YCSBGen::YCSBGeneratorOptions();
    work_options.num_keys = num_keys;
  }
  work_options.max_value_size = options.maxKVSizeBytes - 34;

  options.env = leveldb::Env::Default();
  options.block_cache = leveldb::NewLRUCache(cache_size);
  options.filter_policy = leveldb::NewBloomFilterPolicy(10);
  options.numKeys = num_keys;
  options.numWriteKeys = num_keys;

  std::filesystem::path db_path(arg_db_path);
  auto db_paths = decode_db_paths(arg_db_paths);
  // options.db_paths = decode_db_paths(arg_db_paths);
  // options.statistics = leveldb::CreateDBStatistics();

  // leveldb::BlockBasedTableOptions table_options;
  // table_options.block_cache = leveldb::NewLRUCache(cache_size);
  // table_options.filter_policy.reset(leveldb::NewBloomFilterPolicy(10,
  // false)); options.table_factory.reset(
  //     leveldb::NewBlockBasedTableFactory(table_options));

  options.fd_size = db_paths[0].second;
  leveldb::DB *db;
  if (work_options.load) {
    std::cerr << "Emptying directories\n";
    empty_directory(db_path);
    for (auto &[path, size] : db_paths) {
      empty_directory(path);
    }
    std::cerr << "Creating database\n";
    options.create_if_missing = true;
  }
  auto s = leveldb::DB::Open(options, db_path.string(), &db);
  if (!s.ok()) {
    std::cerr << s.ToString() << std::endl;
    return -1;
  }

  std::string cmd =
      "pidstat -p " + std::to_string(getpid()) +
      " -Hu 1 | awk '{if(NR>3){print $1,$8; fflush(stdout)}}' > " +
      db_path.c_str() + "/cpu &";
  std::cerr << cmd << std::endl;
  std::system(cmd.c_str());

  work_options.db = db;
  work_options.switches = switches;
  work_options.db_path = db_path;
  work_options.enable_fast_process = vm.count("enable_fast_process");
  if (format == "plain") {
    work_options.format_type = FormatType::Plain;
  } else if (format == "plain-length-only") {
    work_options.format_type = FormatType::PlainLengthOnly;
  } else if (format == "ycsb") {
    work_options.format_type = FormatType::YCSB;
  } else {
    rusty_panic("Unrecognized format %s", format.c_str());
  }
  work_options.export_ans_xxh64 = vm.count("export_ans_xxh64");

  Tester tester(work_options);

  std::atomic<bool> should_stop(false);
  std::thread stat_printer(bg_stat_printer, &tester, &should_stop);

  std::thread period_print_thread([&]() {
    std::ofstream period_stats(db_path / "period_stats");
    while (!should_stop.load()) {
      tester.print_other_stats(period_stats);
      std::this_thread::sleep_for(std::chrono::seconds(1));
    }
  });

  tester.Test();
  tester.print_other_stats(std::cerr);

  should_stop.store(true, std::memory_order_relaxed);
  stat_printer.join();
  period_print_thread.join();
  delete db;

  return 0;
}
