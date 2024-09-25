#include <pthread.h>
#include <rocksdb/cache.h>
#include <rocksdb/db.h>
#include <rocksdb/filter_policy.h>
#include <rocksdb/iostats_context.h>
#include <rocksdb/perf_context.h>
#include <rocksdb/rate_limiter.h>
#include <rocksdb/statistics.h>
#include <rocksdb/table.h>
#include <rusty/macro.h>
#include <rusty/sync.h>
#include <rusty/time.h>
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
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <deque>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <limits>
#include <optional>
#include <queue>
#include <set>
#include <string>
#include <thread>
#include <vector>

#include "ycsbgen/ycsbgen.hpp"

std::optional<std::ofstream>& get_key_hit_level_out();

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

static bool has_background_work(rocksdb::DB* db) {
  uint64_t flush_pending;
  uint64_t compaction_pending;
  uint64_t flush_running;
  uint64_t compaction_running;
  bool ok = db->GetIntProperty(
      rocksdb::Slice("rocksdb.mem-table-flush-pending"), &flush_pending);
  rusty_assert(ok);
  ok = db->GetIntProperty(rocksdb::Slice("rocksdb.compaction-pending"),
                          &compaction_pending);
  rusty_assert(ok);
  ok = db->GetIntProperty(rocksdb::Slice("rocksdb.num-running-flushes"),
                          &flush_running);
  rusty_assert(ok);
  ok = db->GetIntProperty(rocksdb::Slice("rocksdb.num-running-compactions"),
                          &compaction_running);
  rusty_assert(ok);
  return flush_pending || compaction_pending || flush_running ||
         compaction_running;
}

static void wait_for_background_work(rocksdb::DB* db) {
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

enum class FormatType {
  Plain,
  PlainLengthOnly,
  YCSB,
};

enum class WorkloadType {
  ConfigFile,
  u24685531,
  hotspot_2_4_6_8,
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
static constexpr uint64_t MASK_COUNT_ACCESS_HOT_PER_TIER = 0x4;
static constexpr uint64_t MASK_KEY_HIT_LEVEL = 0x8;

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
  rocksdb::DB* db;
  const rocksdb::Options* options;
  uint64_t switches;
  std::filesystem::path db_path;
  bool enable_fast_process{false};
  size_t num_threads{1};
  bool enable_fast_generator{false};
  WorkloadType workload_type;
  YCSBGen::YCSBGeneratorOptions ycsb_gen_options;
  double db_paths_soft_size_limit_multiplier;
  std::shared_ptr<rocksdb::RateLimiter> rate_limiter;
  bool export_key_only_trace{false};
  bool export_ans_xxh64{false};
  std::string std_ans_prefix;
};

class Tester {
 public:
  Tester(const WorkOptions& option)
      : options_(option),
        perf_contexts_(options_.num_threads),
        iostats_contexts_(options_.num_threads) {
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
      switch (options_.workload_type) {
        case WorkloadType::ConfigFile:
          GenerateAndExecute(info_json_out);
          break;
        case WorkloadType::u24685531:
          u24685531(info_json_out);
          break;
        case WorkloadType::hotspot_2_4_6_8:
          hotspot_2_4_6_8(info_json_out);
      }
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
      *info_json_out.lock() << "\t\"not-found\": " << not_found << ","
                            << "\t\"scanned-records\": " << scanned << "\n}";
    } else {
      rusty_assert_eq(not_found, (uint64_t)0);
      rusty_assert_eq(scanned, (uint64_t)0);
    }
  }

  void print_other_stats(std::ostream& log) {
    const std::shared_ptr<rocksdb::Statistics>& stats =
        options_.options->statistics;
    log << "Timestamp: " << timestamp_ns() << "\n";
    log << "rocksdb.block.cache.data.miss: "
        << stats->getTickerCount(rocksdb::BLOCK_CACHE_DATA_MISS) << '\n';
    log << "rocksdb.block.cache.data.hit: "
        << stats->getTickerCount(rocksdb::BLOCK_CACHE_DATA_HIT) << '\n';
    log << "rocksdb.bloom.filter.useful: "
        << stats->getTickerCount(rocksdb::BLOOM_FILTER_USEFUL) << '\n';
    log << "rocksdb.bloom.filter.full.positive: "
        << stats->getTickerCount(rocksdb::BLOOM_FILTER_FULL_POSITIVE) << '\n';
    log << "rocksdb.bloom.filter.full.true.positive: "
        << stats->getTickerCount(rocksdb::BLOOM_FILTER_FULL_TRUE_POSITIVE)
        << '\n';
    log << "rocksdb.memtable.hit: "
        << stats->getTickerCount(rocksdb::MEMTABLE_HIT) << '\n';
    log << "rocksdb.l0.hit: " << stats->getTickerCount(rocksdb::GET_HIT_L0)
        << '\n';
    log << "rocksdb.l1.hit: " << stats->getTickerCount(rocksdb::GET_HIT_L1)
        << '\n';
    log << "rocksdb.rocksdb.l2andup.hit: "
        << stats->getTickerCount(rocksdb::GET_HIT_L2_AND_UP) << '\n';
    log << "leader write count: "
        << stats->getTickerCount(rocksdb::LEADER_WRITE_COUNT) << '\n';
    log << "non leader write count: "
        << stats->getTickerCount(rocksdb::NON_LEADER_WRITE_COUNT) << '\n';

    log << "Promotion cache hits: "
        << stats->getTickerCount(rocksdb::PROMOTION_CACHE_GET_HIT) << '\n';
    log << "Promotion cache insert fail to lock: "
        << stats->getTickerCount(rocksdb::PROMOTION_CACHE_INSERT_FAIL_LOCK)
        << '\n';
    log << "Promotion cache insert fail due to compacted: "
        << stats->getTickerCount(rocksdb::PROMOTION_CACHE_INSERT_FAIL_COMPACTED)
        << '\n';
    log << "Promotion cache insert success: "
        << stats->getTickerCount(rocksdb::PROMOTION_CACHE_INSERT) << '\n';

    print_timers(log);

    {
      std::unique_lock lck(thread_local_m_);
      if (perf_contexts_[0]) {
        log << "rocksdb Perf: " << perf_contexts_[0]->ToString() << '\n';
      }
      if (iostats_contexts_[0]) {
        log << "rocksdb IOStats: " << iostats_contexts_[0]->ToString() << '\n';
      }
    }

    log << "stat end===" << std::endl;
  }

 private:
  class Worker {
   public:
    Worker(Tester& tester, size_t id)
        : tester_(tester),
          id_(id),
          options_(tester.options_),
          std_ans_(options_.std_ans_prefix.empty()
                       ? std::nullopt
                       : std::optional<std::ifstream>(options_.std_ans_prefix +
                                                      std::to_string(id_))),
          ans_out_(options_.switches & MASK_OUTPUT_ANS
                       ? std::optional<std::ofstream>(
                             options_.db_path / ("ans_" + std::to_string(id)))
                       : std::nullopt) {
      if (std_ans_.has_value()) {
        rusty_assert(std_ans_.value(), "Fail to open %s",
                     (options_.std_ans_prefix + std::to_string(id_)).c_str());
      }
    }

    void load(YCSBGen::YCSBLoadGenerator& loader) {
      while (!loader.IsEOF()) {
        auto op = loader.GetNextOp();
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

      rocksdb::SetPerfLevel(rocksdb::PerfLevel::kEnableTimeExceptForMutex);
      {
        std::unique_lock lck(tester_.thread_local_m_);
        tester_.perf_contexts_[id_] = rocksdb::get_perf_context();
        tester_.iostats_contexts_[id_] = rocksdb::get_iostats_context();
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

      std::ofstream(options_.db_path / ("run-phase-perf-context-" + id))
          << tester_.perf_contexts_[id_]->ToString();
      std::ofstream(options_.db_path / ("run-phase-iostats-contexts-" + id))
          << tester_.iostats_contexts_[id_]->ToString();
      {
        std::unique_lock lck(tester_.thread_local_m_);
        tester_.perf_contexts_[id_] = nullptr;
        tester_.iostats_contexts_[id_] = nullptr;
      }
    }
    void maybe_enable_key_hit_level() {
      if (options_.switches & MASK_KEY_HIT_LEVEL) {
        get_key_hit_level_out() = std::make_optional<std::ofstream>(
            options_.db_path / ("key-hit-level-" + std::to_string(id_)));
      }
    }
    void run(YCSBGen::YCSBRunGenerator& runner) {
      prepare_run_phase();
      std::mt19937_64 rndgen(id_ + options_.ycsb_gen_options.base_seed);

      maybe_enable_key_hit_level();
      std::optional<std::ofstream> key_only_trace_out =
          options_.export_key_only_trace
              ? std::make_optional<std::ofstream>(
                    options_.db_path /
                    (std::to_string(id_) + "_key_only_trace"))
              : std::nullopt;
      std::string value;
      while (!runner.IsEOF()) {
        auto op = runner.GetNextOp(rndgen);
        if (key_only_trace_out.has_value())
          key_only_trace_out.value()
              << to_string(op.type) << ' ' << op.key << '\n';
        process_op(op, &value);
        tester_.progress_.fetch_add(1, std::memory_order_relaxed);
      }
      finish_run_phase();
    }
    void work(bool run, BlockChannel<YCSBGen::Operation>& chan) {
      if (run) {
        prepare_run_phase();
      }
      maybe_enable_key_hit_level();
      std::string value;
      for (;;) {
        auto block = chan.GetBlock();
        if (block.empty()) {
          break;
        }
        for (const YCSBGen::Operation& op : block) {
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
      auto s =
          options_.db->Put(write_options_, put.key,
                           rocksdb::Slice(put.value.data(), put.value.size()));
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
      s = options_.db->Put(write_options_, op.key,
                           rocksdb::Slice(op.value.data(), op.value.size()));
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
        std::unique_ptr<rocksdb::Iterator> it(
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
          if (std_ans_.has_value()) {
            std::string std_ans;
            std_ans_.value() >> std_ans;
            if (ans != std_ans) {
              std::cerr << "Result:\n"
                        << ans << "\nStandard result:\n"
                        << std_ans << std::endl;
              rusty_panic();
            }
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
    rocksdb::ReadOptions read_options_;
    rocksdb::WriteOptions write_options_;

    uint64_t not_found_{0};
    uint64_t scanned_{0};
    XXH64_state_t* ans_xxhash_state_{nullptr};
    std::optional<std::ifstream> std_ans_;
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
        opblocks[i].Push(
            YCSBGen::Operation(type, std::move(key), std::move(value)));
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
            YCSBGen::Operation(YCSBGen::OpType::DELETE, std::move(key), {}));
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
    std::string rocksdb_stats;
    *info_json_out.lock() << "\t\"load-time(secs)\": "
                          << load_start.elapsed().as_secs_double() << ','
                          << std::endl;
    rusty_assert(options_.db->GetProperty("rocksdb.stats", &rocksdb_stats));
    std::ofstream(options_.db_path / "rocksdb-stats-load.txt") << rocksdb_stats;

    auto load_wait_start = rusty::time::Instant::now();
    wait_for_background_work(options_.db);
    *info_json_out.lock() << "\t\"load-wait-time(secs)\": "
                          << load_wait_start.elapsed().as_secs_double() << ','
                          << std::endl;
    rusty_assert(options_.db->GetProperty("rocksdb.stats", &rocksdb_stats));
    std::ofstream(options_.db_path / "rocksdb-stats-load-finish.txt")
        << rocksdb_stats;

    std::ofstream other_stats_out(options_.db_path /
                                  "other-stats-load-finish.txt");
    print_other_stats(other_stats_out);
  }

  void prepare_run_phase(
      const rusty::sync::Mutex<std::ofstream>& info_json_out) {
    const auto& ts = timers.timers();
    for (const auto& timer : ts) {
      timer.reset();
    }

    if (options_.rate_limiter) {
      options_.rate_limiter->SetBytesPerSecond(
          std::numeric_limits<int64_t>::max());
    }

    options_.db->SetOptions(
        {{"db_paths_soft_size_limit_multiplier",
          std::to_string(options_.db_paths_soft_size_limit_multiplier)}});
    std::cerr << "options.db_paths_soft_size_limit_multiplier: ";
    print_vector(options_.db->GetOptions().db_paths_soft_size_limit_multiplier);
    std::cerr << std::endl;

    *info_json_out.lock() << "\t\"run-start-timestamp(ns)\": " << timestamp_ns()
                          << ',' << std::endl;
  }

  void finish_run_phase(const rusty::sync::Mutex<std::ofstream>& info_json_out,
                        rusty::time::Instant run_start) {
    std::string rocksdb_stats;
    *info_json_out.lock() << "\t\"run-end-timestamp(ns)\": " << timestamp_ns()
                          << ",\n"
                          << "\t\"run-time(secs)\": "
                          << run_start.elapsed().as_secs_double() << ','
                          << std::endl;
    rusty_assert(options_.db->GetProperty("rocksdb.stats", &rocksdb_stats));
    std::ofstream(options_.db_path / "rocksdb-stats-run.txt") << rocksdb_stats;

    auto run_wait_start = rusty::time::Instant::now();
    wait_for_background_work(options_.db);
    *info_json_out.lock() << "\t\"run_wait_time(secs)\": "
                          << run_wait_start.elapsed().as_secs_double() << ","
                          << std::endl;
    rusty_assert(options_.db->GetProperty("rocksdb.stats", &rocksdb_stats));
    std::ofstream(options_.db_path / "rocksdb-stats.txt") << rocksdb_stats;
  }

  void LoadPhase(const rusty::sync::Mutex<std::ofstream>& info_json_out,
                 const YCSBGen::YCSBGeneratorOptions& options) {
    std::vector<std::thread> threads;
    YCSBGen::YCSBLoadGenerator loader(options);
    std::cerr << "Load phase YCSB Options: " << options.ToString() << std::endl;

    *info_json_out.lock() << "\t\"num-load-op\": " << options.record_count
                          << ',' << std::endl;

    auto load_start = rusty::time::Instant::now();
    for (size_t i = 0; i < options_.num_threads; ++i) {
      threads.emplace_back([this, &loader, i]() { workers_[i].load(loader); });
    }
    for (auto& t : threads) t.join();
    finish_load_phase(info_json_out, load_start);
  }

  void GenerateAndExecute(
      const rusty::sync::Mutex<std::ofstream>& info_json_out) {
    if (options_.load) {
      LoadPhase(info_json_out, options_.ycsb_gen_options);
    }

    if (options_.run) {
      std::vector<std::thread> threads;
      prepare_run_phase(info_json_out);
      auto run_start = rusty::time::Instant::now();
      YCSBGen::YCSBRunGenerator runner(options_.ycsb_gen_options,
                                       options_.ycsb_gen_options.record_count);
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

  void RunPhase(const YCSBGen::YCSBGeneratorOptions& options,
                std::unique_ptr<YCSBGen::KeyGenerator> key_generator) {
    std::vector<std::thread> threads;
    YCSBGen::YCSBRunGenerator runner(options, options.record_count,
                                     std::move(key_generator));
    for (size_t i = 0; i < options_.num_threads; ++i) {
      threads.emplace_back([this, &runner, i]() { workers_[i].run(runner); });
    }
    for (auto& t : threads) t.join();
  }
  const uint64_t num_load_keys = 110000000;
  const uint64_t num_run_op = 220000000;
  void run_hotspot(uint64_t offset, double hotspot_set_fraction) {
    RunPhase(
        YCSBGen::YCSBGeneratorOptions{
            .record_count = num_load_keys,
            .operation_count = num_run_op,
        },
        std::make_unique<YCSBGen::HotspotGenerator>(0, num_load_keys, offset,
                                                    hotspot_set_fraction,
                                                    1 - hotspot_set_fraction));
  }
  void load_phase(const rusty::sync::Mutex<std::ofstream>& info_json_out) {
    if (options_.load) {
      LoadPhase(info_json_out, YCSBGen::YCSBGeneratorOptions{
                                   .record_count = num_load_keys,
                                   .operation_count = num_run_op,
                                   .read_proportion = 0,
                                   .insert_proportion = 1,
                               });
    }
  }
  void u24685531(const rusty::sync::Mutex<std::ofstream>& info_json_out) {
    const uint64_t offset = 0.05 * num_load_keys;
    load_phase(info_json_out);
    if (options_.run) {
      prepare_run_phase(info_json_out);
      auto run_start = rusty::time::Instant::now();
      RunPhase(
          YCSBGen::YCSBGeneratorOptions{
              .record_count = num_load_keys,
              .operation_count = num_run_op,
              .request_distribution = "uniform",
          },
          std::make_unique<YCSBGen::UniformGenerator>(0, num_load_keys));
      run_hotspot(0, 0.02);
      run_hotspot(0, 0.04);
      run_hotspot(0, 0.06);
      run_hotspot(0, 0.08);
      run_hotspot(0, 0.05);
      run_hotspot(offset, 0.05);
      run_hotspot(offset, 0.03);
      run_hotspot(offset, 0.01);
      finish_run_phase(info_json_out, run_start);
    }
  }
  void hotspot_2_4_6_8(const rusty::sync::Mutex<std::ofstream>& info_json_out) {
    load_phase(info_json_out);
    if (options_.run) {
      prepare_run_phase(info_json_out);
      auto run_start = rusty::time::Instant::now();
      run_hotspot(0, 0.02);
      run_hotspot(0, 0.04);
      run_hotspot(0, 0.06);
      run_hotspot(0, 0.08);
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

  std::vector<rocksdb::PerfContext*> perf_contexts_;
  std::vector<rocksdb::IOStatsContext*> iostats_contexts_;
  std::mutex thread_local_m_;

  std::atomic<uint64_t> progress_{0};
  std::atomic<uint64_t> progress_get_{0};
};
