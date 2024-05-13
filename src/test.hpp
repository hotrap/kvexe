#include <leveldb/cache.h>
#include <leveldb/db.h>
#include <leveldb/env.h>
#include <leveldb/filter_policy.h>
#include <leveldb/table.h>
#include <rusty/keyword.h>
#include <rusty/macro.h>
#include <rusty/primitive.h>
#include <rusty/sync.h>
#include <rusty/time.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <boost/program_options/errors.hpp>
#include <boost/program_options/options_description.hpp>
#include <boost/program_options/parsers.hpp>
#include <boost/program_options/variables_map.hpp>
#include <cctype>
#include <chrono>
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

enum class OpType {
  INSERT,
  READ,
  UPDATE,
  RMW,
  DELETE,
};
static inline const char* to_string(OpType type) {
  switch (type) {
    case OpType::INSERT:
      return "INSERT";
    case OpType::READ:
      return "READ";
    case OpType::UPDATE:
      return "UPDATE";
    case OpType::RMW:
      return "RMW";
    case OpType::DELETE:
      return "DELETE";
  }
  rusty_panic();
}

enum class TimerType : size_t {
  kPut,
  kGet,
  kDelete,
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
    "Put",       "Get",         "Delete", "InputOperation", "InputInsert",
    "InputRead", "InputUpdate", "Output", "Serialize",      "Deserialize",
};
static_assert(sizeof(timer_names) == TIMER_NUM * sizeof(const char*));
static counter_timer::TypedTimers<TimerType> timers(TIMER_NUM);

static std::atomic<time_t> put_cpu_nanos(0);
static std::atomic<time_t> get_cpu_nanos(0);
static std::atomic<time_t> delete_cpu_nanos(0);

constexpr uint64_t MASK_LATENCY = 0x1;
constexpr uint64_t MASK_OUTPUT_ANS = 0x2;

struct Operation {
  OpType type;
  std::string key;
  std::vector<char> value;

  Operation() {}

  Operation(OpType _type, const std::string& _key,
            const std::vector<char>& _value)
      : type(_type), key(_key), value(_value) {}

  Operation(OpType _type, std::string&& _key, std::vector<char>&& _value)
      : type(_type), key(std::move(_key)), value(std::move(_value)) {}
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
  BlockChannel(size_t limit_size = 64) : limit_size_(limit_size) {}
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
  std::optional<std::filesystem::path> load_trace;
  std::optional<std::filesystem::path> run_trace;
  FormatType format_type;
  leveldb::DB* db;
  uint64_t switches;
  std::filesystem::path db_path;
  std::atomic<uint64_t>* progress;
  std::atomic<uint64_t>* progress_get;
  bool enable_fast_process{false};
  size_t num_threads{1};
  size_t opblock_size{1024};
  bool enable_fast_generator{false};
  YCSBGen::YCSBGeneratorOptions ycsb_gen_options;
  bool export_key_only_trace{false};

  size_t num_keys;  // The number of keys after load phase and run phase.
  size_t num_load_ops;
};

void print_ans(std::ofstream& out, std::string value) { out << value << '\n'; }

void print_latency(std::ofstream& out, OpType op, uint64_t nanos) {
  out << timestamp_ns() << ' ' << to_string(op) << ' ' << nanos << '\n';
}

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
  Tester(const WorkOptions& option) : options_(option) {}

  void Test(const rusty::sync::Mutex<std::ofstream>& info_json_out) {
    for (size_t i = 0; i < options_.num_threads; ++i) {
      workers_.emplace_back(*this, i);
    }
    if (options_.enable_fast_generator) {
      GenerateAndExecute(info_json_out);
    } else {
      ReadAndExecute(info_json_out);
    }
  }

  uint64_t GetOpParseCounts() const { return parse_counts_; }

  uint64_t GetNotFoundCounts() const {
    return notfound_counts_.load(std::memory_order_relaxed);
  }

 private:
  class Worker {
   public:
    Worker(Tester& tester, size_t id)
        : tester_(tester),
          id_(id),
          options_(tester.options_),
          ignore_notfound(options_.enable_fast_process),
          notfound_counts_(tester.notfound_counts_),
          ans_out_(options_.switches & MASK_OUTPUT_ANS
                       ? std::optional<std::ofstream>(
                             options_.db_path / ("ans_" + std::to_string(id)))
                       : std::nullopt) {}

    void load(YCSBGen::YCSBLoadGenerator& loader) {
      Operation op;
      while (!loader.IsEOF()) {
        auto ycsb_op = loader.GetNextOp();
        op.key = gen_prism_key(ycsb_op.key, 0, options_.num_keys);
        op.value = std::move(ycsb_op.value);
        rusty_assert(ycsb_op.type == YCSBGen::OpType::INSERT);
        op.type = OpType::INSERT;
        do_put(op);
        options_.progress->fetch_add(1, std::memory_order_relaxed);
      }
    }
    void prepare_run_phase() {
      if (options_.switches & MASK_LATENCY) {
        latency_out_ = std::make_optional<std::ofstream>(
            options_.db_path / ("latency-" + std::to_string(id_)));
      }
    }
    void run(YCSBGen::YCSBRunGenerator& runner) {
      std::mt19937_64 rndgen(id_ + options_.ycsb_gen_options.base_seed);

      Operation op;
      std::optional<std::ofstream> key_only_trace_out =
          options_.export_key_only_trace
              ? std::make_optional<std::ofstream>(
                    options_.db_path /
                    (std::to_string(id_) + "_key_only_trace"))
              : std::nullopt;
      std::string value(4096, 0);
      while (!runner.IsEOF()) {
        auto ycsb_op = runner.GetNextOp(rndgen);
        op.key = gen_prism_key(ycsb_op.key, 0, options_.num_keys);
        op.value = std::move(ycsb_op.value);
        switch (ycsb_op.type) {
          case YCSBGen::OpType::INSERT:
            op.type = OpType::INSERT;
            break;
          case YCSBGen::OpType::READ:
            op.type = OpType::READ;
            break;
          case YCSBGen::OpType::UPDATE:
            op.type = OpType::UPDATE;
            break;
          case YCSBGen::OpType::RMW:
            op.type = OpType::RMW;
            break;
        }
        if (key_only_trace_out.has_value())
          key_only_trace_out.value()
              << to_string(op.type) << ' ' << op.key << '\n';
        process_op(op, &value);
        uint64_t progress =
            options_.progress->fetch_add(1, std::memory_order_relaxed);
        if (progress == options_.num_load_ops) {
          options_.db->SetDbMode(false);
          options_.db->ResetMigrationStats();
        }
      }
    }
    void work(BlockChannel<Operation>& chan) {
      std::string value;
      for (;;) {
        auto block = chan.GetBlock();
        if (block.empty()) {
          break;
        }
        for (const Operation& op : block) {
          process_op(op, &value);
          options_.progress->fetch_add(1, std::memory_order_relaxed);
        }
      }
    }

   private:
    void do_put(const Operation& put) {
      time_t put_cpu_start = cpu_timestamp_ns();
      auto put_start = rusty::time::Instant::now();
      auto s = options_.db->Put(
          write_options_, put.key,
          leveldb::Slice(put.value.data(),
                         std::min<size_t>(990, put.value.size())));
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
    bool do_read(const Operation& read, std::string* value) {
      time_t get_cpu_start = cpu_timestamp_ns();
      auto get_start = rusty::time::Instant::now();
      auto s = options_.db->Get(read_options_, read.key, value);
      auto get_time = get_start.elapsed();
      time_t get_cpu_ns = cpu_timestamp_ns() - get_cpu_start;
      if (!s.ok()) {
        if (s.IsNotFound() && ignore_notfound) {
          return false;
        } else {
          std::string err = s.ToString();
          rusty_panic("GET failed with error: %s\n", err.c_str());
        }
      }
      timers.timer(TimerType::kGet).add(get_time);
      get_cpu_nanos.fetch_add(get_cpu_ns, std::memory_order_relaxed);
      if (latency_out_) {
        print_latency(latency_out_.value(), OpType::READ, get_time.as_nanos());
      }
      options_.progress_get->fetch_add(1, std::memory_order_relaxed);
      return true;
    }

    void do_read_modify_write(const Operation& op) {
      time_t get_cpu_start = cpu_timestamp_ns();
      auto start = rusty::time::Instant::now();
      std::string value;
      auto s = options_.db->Get(read_options_, op.key, &value);
      if (!s.ok()) {
        if (s.IsNotFound() && ignore_notfound) {
          value = "";
        } else {
          std::string err = s.ToString();
          rusty_panic("GET failed with error: %s\n", err.c_str());
        }
      }
      time_t put_cpu_start = cpu_timestamp_ns();
      time_t get_cpu_ns = put_cpu_start - get_cpu_start;
      s = options_.db->Put(
          write_options_, op.key,
          leveldb::Slice(op.value.data(),
                         std::min<size_t>(990, op.value.size())));
      time_t put_cpu_ns = cpu_timestamp_ns() - put_cpu_start;
      if (!s.ok()) {
        std::string err = s.ToString();
        rusty_panic("Update failed with error: %s\n", err.c_str());
      }
      get_cpu_nanos.fetch_add(get_cpu_ns, std::memory_order_relaxed);
      put_cpu_nanos.fetch_add(put_cpu_ns, std::memory_order_relaxed);
      if (latency_out_) {
        print_latency(latency_out_.value(), OpType::RMW,
                      start.elapsed().as_nanos());
      }
      options_.progress_get->fetch_add(1, std::memory_order_relaxed);
    }

    void do_delete(const Operation& op) {
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
        print_latency(latency_out_.value(), OpType::DELETE, time.as_nanos());
      }
    }

    void process_op(const Operation& op, std::string* value) {
      switch (op.type) {
        case OpType::INSERT:
        case OpType::UPDATE:
          do_put(op);
          break;
        case OpType::READ: {
          bool found = do_read(op, value);
          if (ans_out_) {
            if (found) {
              print_ans(ans_out_.value(), *value);
            } else {
              print_ans(ans_out_.value(), "");
            }
          }
          if (!found) {
            local_notfound_counts++;
            if ((local_read_progress & 15) == 15) {
              notfound_counts_ += local_notfound_counts;
              local_notfound_counts = 0;
            }
          }
          local_read_progress++;
        } break;
        case OpType::RMW:
          do_read_modify_write(op);
          break;
        case OpType::DELETE:
          do_delete(op);
          break;
      }
    }

    Tester& tester_;
    size_t id_;
    const WorkOptions& options_;
    leveldb::ReadOptions read_options_;
    leveldb::WriteOptions write_options_;
    bool ignore_notfound{false};
    std::atomic<uint64_t>& notfound_counts_;

    uint64_t local_notfound_counts{0};
    uint64_t local_read_progress{0};
    std::optional<std::ofstream> ans_out_;
    std::optional<std::ofstream> latency_out_;
  };

  void parse(std::istream& trace) {
    size_t num_channels =
        options_.enable_fast_process ? 1 : options_.num_threads;
    std::vector<BlockChannel<Operation>> channel_for_workers(num_channels);

    std::vector<BlockChannelClient<Operation>> opblocks;
    for (auto& channel : channel_for_workers) {
      opblocks.emplace_back(&channel, options_.opblock_size);
    }

    std::vector<std::thread> threads;
    for (size_t i = 0; i < options_.num_threads; i++) {
      threads.emplace_back([this, &channel_for_workers, i]() {
        size_t index = options_.enable_fast_process ? 0 : i;
        workers_[i].work(channel_for_workers[index]);
      });
    }

    std::hash<std::string> hasher{};

    size_t key_num = options_.num_keys;
    size_t key_id_pre = 0;

    while (1) {
      std::string op;
      trace >> op;
      if (!trace) {
        break;
      }
      if (op == "INSERT") {
        if (options_.format_type == FormatType::YCSB) {
          handle_table_name(trace);
        }
        std::string key;
        trace >> key;
        int i = options_.enable_fast_process
                    ? 0
                    : hasher(key) % options_.num_threads;
        if (options_.format_type == FormatType::YCSB) {
          opblocks[i].Push(Operation(OpType::INSERT,
                                     gen_prism_key(key, key_id_pre, key_num),
                                     read_value(trace)));
        } else {
          rusty_assert_eq(trace.get(), ' ');
          char c;
          std::vector<char> value;
          if (options_.format_type == FormatType::Plain) {
            while ((c = trace.get()) != '\n' && c != EOF) {
              value.push_back(c);
            }
          } else {
            size_t value_length;
            trace >> value_length;
            value.resize(value_length);
            size_t copied = std::min(key.size(), value_length);
            memcpy(value.data(), key.data(), copied);
            memset(value.data() + copied, '0', value_length - copied);
          }
          opblocks[i].Push(Operation(OpType::INSERT,
                                     gen_prism_key(key, key_id_pre, key_num),
                                     std::move(value)));
        }
        parse_counts_++;
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
        opblocks[i].Push(Operation(
            OpType::READ, gen_prism_key(key, key_id_pre, key_num), {}));
        parse_counts_++;
      } else if (op == "UPDATE") {
        if (options_.format_type == FormatType::Plain) {
          rusty_panic("UPDATE in plain format is not supported yet\n");
        }
        handle_table_name(trace);
        std::string key;
        trace >> key;
        int i = options_.enable_fast_process
                    ? 0
                    : hasher(key) % options_.num_threads;
        opblocks[i].Push(Operation(OpType::UPDATE,
                                   gen_prism_key(key, key_id_pre, key_num),
                                   read_value(trace)));
        parse_counts_++;
      } else if (op == "DELETE") {
        std::string key;
        rusty_assert(options_.format_type == FormatType::Plain ||
                     options_.format_type == FormatType::PlainLengthOnly);
        trace >> key;
        int i = options_.enable_fast_process
                    ? 0
                    : hasher(key) % options_.num_threads;
        opblocks[i].Push(Operation(
            OpType::DELETE, gen_prism_key(key, key_id_pre, key_num), {}));
        parse_counts_++;
      } else {
        std::cerr << "Ignore line: " << op;
        std::getline(trace, op);  // Skip the rest of the line
        std::cerr << op << std::endl;
      }
    }

    for (auto& o : opblocks) {
      o.Flush();
      o.Finish();
    }

    for (auto& t : threads) t.join();
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
  }

  void prepare_run_phase(
      const rusty::sync::Mutex<std::ofstream>& info_json_out) {
    for (auto& worker : workers_) {
      worker.prepare_run_phase();
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
      if (options_.load_trace.has_value()) {
        trace_file = std::ifstream(options_.load_trace.value());
        rusty_assert(trace_file.value());
      }
      std::istream& trace =
          trace_file.has_value() ? trace_file.value() : std::cin;

      auto start = rusty::time::Instant::now();
      parse(trace);
      finish_load_phase(info_json_out, start);
    }
    if (options_.run) {
      std::optional<std::ifstream> trace_file;
      if (options_.run_trace.has_value()) {
        trace_file = std::ifstream(options_.run_trace.value());
        rusty_assert(trace_file.value());
      }
      std::istream& trace =
          trace_file.has_value() ? trace_file.value() : std::cin;

      prepare_run_phase(info_json_out);
      auto start = rusty::time::Instant::now();
      parse(trace);
      finish_run_phase(info_json_out, start);
    }
  }

  WorkOptions options_;
  std::vector<Worker> workers_;

  uint64_t parse_counts_{0};
  std::atomic<uint64_t> notfound_counts_{0};

  friend class Worker;
};
