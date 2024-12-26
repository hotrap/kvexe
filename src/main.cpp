#include <autotuner.h>
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
#include <counter_timer_vec.hpp>
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

#include "ralt.h"
#include "rocksdb/advanced_cache.h"
#include "rocksdb/ralt.h"
#include "ycsbgen/ycsbgen.hpp"

thread_local std::optional<std::ofstream> key_hit_level_out;
std::optional<std::ofstream> &get_key_hit_level_out() {
  return key_hit_level_out;
}

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
void print_vector(const std::vector<T> &v) {
  std::cerr << '[';
  for (double x : v) {
    std::cerr << x << ',';
  }
  std::cerr << "]";
}

static bool has_background_work(rocksdb::DB *db) {
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

static void wait_for_background_work(rocksdb::DB *db) {
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
  hotspot_5_shift_5,
};

static inline const char *to_string(YCSBGen::OpType type) {
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
static inline void print_latency(std::ofstream &out, YCSBGen::OpType op,
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
static const char *timer_names[] = {
    "Put",         "Get",       "Delete",      "Scan",   "InputOperation",
    "InputInsert", "InputRead", "InputUpdate", "Output", "Serialize",
    "Deserialize",
};
static_assert(sizeof(timer_names) == TIMER_NUM * sizeof(const char *));
static counter_timer::TypedTimers<TimerType> timers(TIMER_NUM);

static inline void print_timers(std::ostream &out) {
  const auto &ts = timers.timers();
  size_t num_types = ts.size();
  for (size_t i = 0; i < num_types; ++i) {
    const auto &timer = ts[i];
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

  void PutBlock(std::vector<T> &&block) {
    std::unique_lock lck(m_);
    q_.push(std::move(block));
    if (reader_waiting_) {
      cv_r_.notify_one();
    }
  }

  void PutBlock(const std::vector<T> &block) {
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
  BlockChannel<T> *channel_;
  std::vector<T> opblock_;
  size_t opnum_{0};

 public:
  BlockChannelClient(BlockChannel<T> *channel, size_t block_size)
      : channel_(channel), opblock_(block_size) {}

  void Push(T &&data) {
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
  rocksdb::DB *db;
  const rocksdb::Options *options;
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

  // For stats
  std::shared_ptr<rocksdb::Cache> block_cache;
};

class Tester {
 public:
  Tester(const WorkOptions &option)
      : options_(option),
        perf_contexts_(options_.num_threads),
        iostats_contexts_(options_.num_threads) {
    for (size_t i = 0; i < options_.num_threads; ++i) {
      workers_.emplace_back(*this, i);
    }
  }

  const WorkOptions &work_options() const { return options_; }
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
      if (options_.workload_type == WorkloadType::ConfigFile) {
        GenerateAndExecute(info_json_out);
      } else {
        load_phase(info_json_out);
        if (options_.run) {
          prepare_run_phase(info_json_out);
          auto run_start = rusty::time::Instant::now();
          switch (options_.workload_type) {
            case WorkloadType::u24685531:
              u24685531();
              break;
            case WorkloadType::hotspot_2_4_6_8:
              hotspot_2_4_6_8();
              break;
            case WorkloadType::hotspot_5_shift_5:
              hotspot_5_shift_5();
              break;
            default:
              rusty_panic();
          }
          finish_run_phase(info_json_out, run_start);
        }
      }
    } else {
      ReadAndExecute(info_json_out);
    }

    uint64_t not_found = 0;
    uint64_t scanned = 0;
    for (const auto &worker : workers_) {
      not_found += worker.not_found();
      scanned += worker.scanned();
    }
    if (options_.run) {
      const rocksdb::Statistics &stats = *options_.options->statistics;
      *info_json_out.lock()
          << "\t\"pc-insert-fail-lock\": "
          << stats.getTickerCount(rocksdb::PROMOTION_CACHE_INSERT_FAIL_LOCK)
          << ",\n"
          << "\t\"pc-insert-fail-compacted\": "
          << stats.getTickerCount(
                 rocksdb::PROMOTION_CACHE_INSERT_FAIL_COMPACTED)
          << ",\n"
          << "\t\"pc-insert\": "
          << stats.getTickerCount(rocksdb::PROMOTION_CACHE_INSERT) << ",\n"
          << "\t\"not-found\": " << not_found << ",\n"
          << "\t\"scanned-records\": " << scanned << "\n}";
    } else {
      rusty_assert_eq(not_found, (uint64_t)0);
      rusty_assert_eq(scanned, (uint64_t)0);
    }
  }

  void print_other_stats(std::ostream &log) {
    const std::shared_ptr<rocksdb::Statistics> &stats =
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

    print_timers(log);

    rocksdb::Cache &block_cache = *work_options().block_cache;
    log << "Block cache usage: " << block_cache.GetUsage()
        << ", pinned usage: " << block_cache.GetPinnedUsage() << '\n';

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
    Worker(Tester &tester, size_t id)
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

    void load(YCSBGen::YCSBLoadGenerator &loader) {
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
    void run(YCSBGen::YCSBRunGenerator &runner) {
      prepare_run_phase();
      std::mt19937_64 rndgen(id_ + options_.ycsb_gen_options.base_seed);

      maybe_enable_key_hit_level();
      std::optional<std::ofstream> key_only_trace_out =
          options_.export_key_only_trace
              ? std::make_optional<std::ofstream>(
                    options_.db_path /
                    (std::to_string(id_) + "_key_only_trace"))
              : std::nullopt;
      while (!runner.IsEOF()) {
        auto op = runner.GetNextOp(rndgen);
        if (key_only_trace_out.has_value())
          key_only_trace_out.value()
              << to_string(op.type) << ' ' << op.key << '\n';
        process_op(op);
        tester_.progress_.fetch_add(1, std::memory_order_relaxed);
      }
      finish_run_phase();
    }
    void work(bool run, BlockChannel<YCSBGen::Operation> &chan) {
      if (run) {
        prepare_run_phase();
      }
      maybe_enable_key_hit_level();
      for (;;) {
        auto block = chan.GetBlock();
        if (block.empty()) {
          break;
        }
        for (const YCSBGen::Operation &op : block) {
          process_op(op);
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
    void do_put(const YCSBGen::Operation &put) {
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
    bool do_read(const YCSBGen::Operation &read,
                 rocksdb::PinnableSlice *value) {
      time_t get_cpu_start = cpu_timestamp_ns();
      auto get_start = rusty::time::Instant::now();
      auto s = options_.db->Get(
          read_options_, options_.db->DefaultColumnFamily(), read.key, value);
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

    void do_read_modify_write(const YCSBGen::Operation &op) {
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

    void do_delete(const YCSBGen::Operation &op) {
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

    void do_scan(const YCSBGen::Operation &op) {
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

    void process_op(const YCSBGen::Operation &op) {
      switch (op.type) {
        case YCSBGen::OpType::INSERT:
        case YCSBGen::OpType::UPDATE:
          do_put(op);
          break;
        case YCSBGen::OpType::READ: {
          rocksdb::PinnableSlice value;
          bool found = do_read(op, &value);
          std::string_view ans;
          if (found) {
            ans = std::string_view(value.data(), value.size());
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

    Tester &tester_;
    size_t id_;
    const WorkOptions &options_;
    rocksdb::ReadOptions read_options_;
    rocksdb::WriteOptions write_options_;

    uint64_t not_found_{0};
    uint64_t scanned_{0};
    XXH64_state_t *ans_xxhash_state_{nullptr};
    std::optional<std::ifstream> std_ans_;
    std::optional<std::ofstream> ans_out_;
    std::optional<std::ofstream> latency_out_;
  };

  void parse(bool run, std::istream &trace) {
    size_t num_channels =
        options_.enable_fast_process ? 1 : options_.num_threads;
    std::vector<BlockChannel<YCSBGen::Operation>> channel_for_workers(
        num_channels);

    std::vector<BlockChannelClient<YCSBGen::Operation>> opblocks;
    for (auto &channel : channel_for_workers) {
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

    for (auto &o : opblocks) {
      o.Flush();
      o.Finish();
    }

    for (auto &t : threads) t.join();

    uint64_t queue_empty_when_put = 0;
    uint64_t queue_non_empty_when_put = 0;
    uint64_t reader_blocked = 0;
    uint64_t reader_not_blocked = 0;
    for (const auto &channel : channel_for_workers) {
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

  void finish_load_phase(const rusty::sync::Mutex<std::ofstream> &info_json_out,
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
      const rusty::sync::Mutex<std::ofstream> &info_json_out) {
    const auto &ts = timers.timers();
    for (const auto &timer : ts) {
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

  void finish_run_phase(const rusty::sync::Mutex<std::ofstream> &info_json_out,
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

  void LoadPhase(const rusty::sync::Mutex<std::ofstream> &info_json_out,
                 const YCSBGen::YCSBGeneratorOptions &options) {
    std::vector<std::thread> threads;
    YCSBGen::YCSBLoadGenerator loader(options);
    std::cerr << "Load phase YCSB Options: " << options.ToString() << std::endl;

    *info_json_out.lock() << "\t\"num-load-op\": " << options.record_count
                          << ',' << std::endl;

    auto load_start = rusty::time::Instant::now();
    for (size_t i = 0; i < options_.num_threads; ++i) {
      threads.emplace_back([this, &loader, i]() { workers_[i].load(loader); });
    }
    for (auto &t : threads) t.join();
    finish_load_phase(info_json_out, load_start);
  }

  void GenerateAndExecute(
      const rusty::sync::Mutex<std::ofstream> &info_json_out) {
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
      for (auto &t : threads) t.join();
      finish_run_phase(info_json_out, run_start);
    }
  }

  void RunPhase(const YCSBGen::YCSBGeneratorOptions &options,
                std::unique_ptr<YCSBGen::KeyGenerator> key_generator) {
    std::vector<std::thread> threads;
    YCSBGen::YCSBRunGenerator runner(options, options.record_count,
                                     std::move(key_generator));
    for (size_t i = 0; i < options_.num_threads; ++i) {
      threads.emplace_back([this, &runner, i]() { workers_[i].run(runner); });
    }
    for (auto &t : threads) t.join();
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
  void load_phase(const rusty::sync::Mutex<std::ofstream> &info_json_out) {
    if (options_.load) {
      LoadPhase(info_json_out, YCSBGen::YCSBGeneratorOptions{
                                   .record_count = num_load_keys,
                                   .operation_count = num_run_op,
                                   .read_proportion = 0,
                                   .insert_proportion = 1,
                               });
    }
  }
  void u24685531() {
    const uint64_t offset = 0.08 * num_load_keys;
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
  }
  void hotspot_2_4_6_8() {
    run_hotspot(0, 0.02);
    run_hotspot(0, 0.04);
    run_hotspot(0, 0.06);
    run_hotspot(0, 0.08);
  }
  void hotspot_5_shift_5() {
    run_hotspot(0, 0.05);
    run_hotspot(0.05 * num_load_keys, 0.05);
  }

  void ReadAndExecute(const rusty::sync::Mutex<std::ofstream> &info_json_out) {
    if (options_.load) {
      std::optional<std::ifstream> trace_file;
      if (!options_.load_trace.empty()) {
        trace_file = std::ifstream(options_.load_trace);
        rusty_assert(trace_file.value());
      }
      std::istream &trace =
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
      std::istream &trace =
          trace_file.has_value() ? trace_file.value() : std::cin;

      prepare_run_phase(info_json_out);
      auto start = rusty::time::Instant::now();
      parse(true, trace);
      finish_run_phase(info_json_out, start);
    }
  }

  WorkOptions options_;
  std::vector<Worker> workers_;

  std::vector<rocksdb::PerfContext *> perf_contexts_;
  std::vector<rocksdb::IOStatsContext *> iostats_contexts_;
  std::mutex thread_local_m_;

  std::atomic<uint64_t> progress_{0};
  std::atomic<uint64_t> progress_get_{0};
};

static inline void empty_directory(std::filesystem::path dir_path) {
  for (auto &path : std::filesystem::directory_iterator(dir_path)) {
    std::filesystem::remove_all(path);
  }
}

constexpr size_t MAX_NUM_LEVELS = 8;

std::vector<rocksdb::DbPath> decode_db_paths(std::string db_paths) {
  std::istringstream in(db_paths);
  std::vector<rocksdb::DbPath> ret;
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

class RaltWrapper : public ralt::RALT {
 public:
  RaltWrapper(const ralt::Options &options, const rocksdb::Comparator *ucmp,
              std::filesystem::path dir, int tier0_last_level,
              size_t init_hot_set_size, size_t max_ralt_size, uint64_t switches,
              size_t max_hot_set_size, size_t min_hot_set_size,
              uint64_t accessed_size_to_decr_counter)
      : ralt::RALT(options, ucmp, dir.c_str(), init_hot_set_size,
                   max_hot_set_size, min_hot_set_size, max_ralt_size,
                   accessed_size_to_decr_counter),
        switches_(switches),
        tier0_last_level_(tier0_last_level),
        count_access_hot_per_tier_{0, 0},
        count_access_fd_hot_(0),
        count_access_fd_cold_(0) {
    for (size_t i = 0; i < MAX_NUM_LEVELS; ++i) {
      level_hits_[i].store(0, std::memory_order_relaxed);
    }
  }
  const char *Name() const override { return "RALT-wrapper"; }
  void HitLevel(int level, rocksdb::Slice key) override {
    if (get_key_hit_level_out().has_value()) {
      get_key_hit_level_out().value()
          << timestamp_ns() << ' ' << key.ToString() << ' ' << level << '\n';
    }
    if (level < 0) level = 0;
    rusty_assert((size_t)level < MAX_NUM_LEVELS);
    level_hits_[level].fetch_add(1, std::memory_order_relaxed);

    if (switches_ & MASK_COUNT_ACCESS_HOT_PER_TIER) {
      size_t tier = level <= tier0_last_level_ ? 0 : 1;
      bool is_hot = IsHot(key);
      if (is_hot)
        count_access_hot_per_tier_[tier].fetch_add(1,
                                                   std::memory_order_relaxed);
      if (tier == 0) {
        if (is_hot) {
          count_access_fd_hot_.fetch_add(1, std::memory_order_relaxed);
        } else {
          count_access_fd_cold_.fetch_add(1, std::memory_order_relaxed);
        }
      }
    }
  }

  std::vector<size_t> hit_tier_count() {
    std::vector<size_t> ret(2, 0);
    size_t tier1_first_level =
        std::min((size_t)(tier0_last_level_ + 1), MAX_NUM_LEVELS);
    size_t i = 0;
    for (; i < tier1_first_level; ++i) {
      ret[0] += level_hits_[i].load(std::memory_order_relaxed);
    }
    for (; i < MAX_NUM_LEVELS; ++i) {
      ret[1] += level_hits_[i].load(std::memory_order_relaxed);
    }
    return ret;
  }

  std::vector<size_t> level_hits() {
    size_t last_level = MAX_NUM_LEVELS;
    size_t last_level_hits;
    do {
      last_level -= 1;
      last_level_hits = level_hits_[last_level].load(std::memory_order_relaxed);
      if (last_level_hits != 0) break;
    } while (last_level > 0);
    std::vector<size_t> ret;
    ret.reserve(last_level + 1);
    for (size_t i = 0; i < last_level; ++i) {
      ret.push_back(level_hits_[i].load(std::memory_order_relaxed));
    }
    ret.push_back(last_level_hits);
    return ret;
  }

  std::vector<uint64_t> hit_hot_count() {
    std::vector<uint64_t> ret;
    for (size_t i = 0; i < 2; ++i)
      ret.push_back(
          count_access_hot_per_tier_[i].load(std::memory_order_relaxed));
    return ret;
  }
  uint64_t count_access_fd_hot() const {
    return count_access_fd_hot_.load(std::memory_order_relaxed);
  }
  uint64_t count_access_fd_cold() const {
    return count_access_fd_cold_.load(std::memory_order_relaxed);
  }

 private:
  const uint64_t switches_;
  int tier0_last_level_;

  std::atomic<uint64_t> level_hits_[MAX_NUM_LEVELS];
  std::atomic<uint64_t> count_access_hot_per_tier_[2];
  std::atomic<uint64_t> count_access_fd_hot_;
  std::atomic<uint64_t> count_access_fd_cold_;
};

void bg_stat_printer(Tester *tester, std::atomic<bool> *should_stop) {
  const WorkOptions &work_options = tester->work_options();
  rocksdb::DB *db = work_options.db;
  const std::filesystem::path &db_path = work_options.db_path;
  const rocksdb::Options *options = work_options.options;
  auto &ralt = *static_cast<RaltWrapper *>(options->ralt.get());

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

  std::ofstream compaction_stats_out(db_path / "compaction-stats");

  std::ofstream timers_out(db_path / "timers");
  timers_out << "Timestamp(ns) compaction-cpu-micros put-cpu-nanos "
                "get-cpu-nanos delete-cpu-nanos";
  uint64_t value;
  bool has_ralt_compaction_thread_cpu_nanos =
      ralt.GetIntProperty("ralt.compaction.thread.cpu.nanos", &value);
  if (has_ralt_compaction_thread_cpu_nanos) {
    timers_out << " ralt.compaction.thread.cpu.nanos";
  }
  bool has_ralt_flush_thread_cpu_nanos =
      ralt.GetIntProperty("ralt.flush.thread.cpu.nanos", &value);
  if (has_ralt_flush_thread_cpu_nanos) {
    timers_out << " ralt.flush.thread.cpu.nanos";
  }
  bool has_ralt_decay_thread_cpu_nanos =
      ralt.GetIntProperty("ralt.decay.thread.cpu.nanos", &value);
  if (has_ralt_decay_thread_cpu_nanos) {
    timers_out << " ralt.decay.thread.cpu.nanos";
  }
  bool has_ralt_compaction_cpu_nanos =
      ralt.GetIntProperty("ralt.compaction.cpu.nanos", &value);
  if (has_ralt_compaction_cpu_nanos) {
    timers_out << " ralt.compaction.cpu.nanos";
  }
  bool has_ralt_flush_cpu_nanos =
      ralt.GetIntProperty("ralt.flush.cpu.nanos", &value);
  if (has_ralt_flush_cpu_nanos) {
    timers_out << " ralt.flush.cpu.nanos";
  }
  bool has_ralt_decay_scan_cpu_nanos =
      ralt.GetIntProperty("ralt.decay.scan.cpu.nanos", &value);
  if (has_ralt_decay_scan_cpu_nanos) {
    timers_out << " ralt.decay.scan.cpu.nanos";
  }
  bool has_ralt_decay_write_cpu_nanos =
      ralt.GetIntProperty("ralt.decay.write.cpu.nanos", &value);
  if (has_ralt_decay_write_cpu_nanos) {
    timers_out << " ralt.decay.write.cpu.nanos";
  }
  timers_out << std::endl;

  std::ofstream rand_read_bytes_out(db_path / "rand-read-bytes");

  // Stats of hotrap

  std::ofstream promoted_or_retained_out(db_path /
                                         "promoted-or-retained-bytes");
  promoted_or_retained_out
      << "Timestamp(ns) by-flush 2fdlast 2sdfront retained.fd retained.sd\n";

  std::ofstream not_promoted_bytes_out(db_path / "not-promoted-bytes");
  not_promoted_bytes_out << "Timestamp(ns) not-hot has-newer-version\n";

  std::ofstream num_accesses_out(db_path / "num-accesses");

  // Stats of RALT

  std::ofstream ralt_io_out(db_path / "ralt-io");
  ralt_io_out << "Timestamp(ns) read write\n";

  std::ofstream ralt_sizes(db_path / "ralt-sizes");
  ralt_sizes << "Timestamp(ns) real-phy-size real-hot-size\n";

  std::ofstream vc_param_out(db_path / "vc_param");
  vc_param_out << "Timestamp(ns) hot-set-size-limit phy-size-limit\n";

  auto stats = options->statistics;

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

    std::string compaction_stats;
    rusty_assert(db->GetProperty(rocksdb::DB::Properties::kCompactionStats,
                                 &compaction_stats));
    compaction_stats_out << "Timestamp(ns) " << timestamp << '\n'
                         << compaction_stats << std::endl;

    uint64_t compaction_cpu_micros;
    rusty_assert(db->GetIntProperty(
        rocksdb::DB::Properties::kCompactionCPUMicros, &compaction_cpu_micros));
    timers_out << timestamp << ' ' << compaction_cpu_micros << ' '
               << put_cpu_nanos.load(std::memory_order_relaxed) << ' '
               << get_cpu_nanos.load(std::memory_order_relaxed) << ' '
               << delete_cpu_nanos.load(std::memory_order_relaxed);
    if (has_ralt_compaction_thread_cpu_nanos) {
      rusty_assert(
          ralt.GetIntProperty("ralt.compaction.thread.cpu.nanos", &value));
      timers_out << ' ' << value;
    }
    if (has_ralt_flush_thread_cpu_nanos) {
      rusty_assert(ralt.GetIntProperty("ralt.flush.thread.cpu.nanos", &value));
      timers_out << ' ' << value;
    }
    if (has_ralt_decay_thread_cpu_nanos) {
      rusty_assert(ralt.GetIntProperty("ralt.decay.thread.cpu.nanos", &value));
      timers_out << ' ' << value;
    }
    if (has_ralt_compaction_cpu_nanos) {
      rusty_assert(ralt.GetIntProperty("ralt.compaction.cpu.nanos", &value));
      timers_out << ' ' << value;
    }
    if (has_ralt_flush_cpu_nanos) {
      rusty_assert(ralt.GetIntProperty("ralt.flush.cpu.nanos", &value));
      timers_out << ' ' << value;
    }
    if (has_ralt_decay_scan_cpu_nanos) {
      rusty_assert(ralt.GetIntProperty("ralt.decay.scan.cpu.nanos", &value));
      timers_out << ' ' << value;
    }
    if (has_ralt_decay_write_cpu_nanos) {
      rusty_assert(ralt.GetIntProperty("ralt.decay.write.cpu.nanos", &value));
      timers_out << ' ' << value;
    }
    timers_out << std::endl;

    std::string rand_read_bytes;
    rusty_assert(db->GetProperty(rocksdb::DB::Properties::kRandReadBytes,
                                 &rand_read_bytes));
    rand_read_bytes_out << timestamp << ' ' << rand_read_bytes << std::endl;

    promoted_or_retained_out
        << timestamp << ' '
        << stats->getTickerCount(rocksdb::PROMOTED_FLUSH_BYTES) << ' '
        << stats->getTickerCount(rocksdb::PROMOTED_2FDLAST_BYTES) << ' '
        << stats->getTickerCount(rocksdb::PROMOTED_2SDFRONT_BYTES) << ' '
        << stats->getTickerCount(rocksdb::RETAINED_FD_BYTES) << ' '
        << stats->getTickerCount(rocksdb::RETAINED_SD_BYTES) << std::endl;

    not_promoted_bytes_out
        << timestamp << ' '
        << stats->getTickerCount(rocksdb::ACCESSED_COLD_BYTES) << ' '
        << stats->getTickerCount(rocksdb::HAS_NEWER_VERSION_BYTES) << std::endl;

    num_accesses_out << timestamp;
    auto level_hits = ralt.level_hits();
    for (size_t hits : level_hits) {
      num_accesses_out << ' ' << hits;
    }
    num_accesses_out << std::endl;

    uint64_t ralt_read;
    rusty_assert(
        ralt.GetIntProperty(ralt::RALT::Properties::kReadBytes, &ralt_read));
    uint64_t ralt_write;
    rusty_assert(
        ralt.GetIntProperty(ralt::RALT::Properties::kWriteBytes, &ralt_write));
    ralt_io_out << timestamp << ' ' << ralt_read << ' ' << ralt_write
                << std::endl;

    ralt_sizes << timestamp << ' ' << ralt.GetRealPhySize() << ' '
               << ralt.GetRealHotSetSize() << std::endl;

    vc_param_out << timestamp << ' ' << ralt.GetHotSetSizeLimit() << ' '
                 << ralt.GetPhySizeLimit() << std::endl;

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

  rocksdb::BlockBasedTableOptions table_options;
  rocksdb::Options options;
  ralt::Options ralt_options;
  WorkOptions work_options;

  namespace po = boost::program_options;
  po::options_description desc("Available options");
  std::string format;
  std::string arg_switches;

  std::string arg_db_path;
  std::string arg_db_paths;
  std::string ralt_path_str;
  size_t cache_size;
  int64_t load_phase_rate_limit;

  double arg_max_hot_set_size;
  double arg_max_ralt_size;
  int compaction_pri;

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
                     "0x2: Output the result of READ\n"
                     "0x4: count access hot per tier\n"
                     "0x8: Log key and the level hit");
  desc.add_options()("num_threads",
                     po::value(&work_options.num_threads)->default_value(1),
                     "The number of threads to execute the trace\n");
  desc.add_options()("enable_fast_process",
                     "Enable fast process including ignoring kNotFound and "
                     "pushing operations in one channel.");
  desc.add_options()("enable_fast_generator", "Enable fast generator");
  desc.add_options()("workload",
                     po::value<std::string>()->default_value("file"),
                     "file/u24685531/2-4-6-8");
  desc.add_options()("workload_file", po::value<std::string>(),
                     "Workload file used in built-in generator");
  desc.add_options()("export_key_only_trace",
                     "Export key-only trace generated by built-in generator.");
  desc.add_options()("export_ans_xxh64", "Export xxhash of ans");
  desc.add_options()("std_ans_prefix",
                     po::value<std::string>(&work_options.std_ans_prefix),
                     "Prefix of standard ans files");

  // Options of rocksdb
  desc.add_options()("max_background_jobs",
                     po::value(&options.max_background_jobs)->default_value(6),
                     "");
  desc.add_options()("level0_file_num_compaction_trigger",
                     po::value(&options.level0_file_num_compaction_trigger),
                     "Number of files in level-0 when compactions start");
  desc.add_options()("use_direct_reads",
                     po::value(&options.use_direct_reads)->default_value(true),
                     "");
  desc.add_options()("use_direct_io_for_flush_and_compaction",
                     po::value(&options.use_direct_io_for_flush_and_compaction)
                         ->default_value(true),
                     "");
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
      "block_size",
      po::value<size_t>(&table_options.block_size)->default_value(16384));
  desc.add_options()("max_bytes_for_level_base",
                     po::value(&options.max_bytes_for_level_base));
  desc.add_options()("optimize_filters_for_hits",
                     "Do not build filters for the last level");
  desc.add_options()("load_phase_rate_limit",
                     po::value(&load_phase_rate_limit)->default_value(0),
                     "0 means not limited.");
  desc.add_options()(
      "db_paths_soft_size_limit_multiplier",
      po::value<double>(&work_options.db_paths_soft_size_limit_multiplier)
          ->default_value(1.1));

  // Options for hotrap
  desc.add_options()("max_hot_set_size",
                     po::value<double>(&arg_max_hot_set_size)->required(),
                     "Max hot set size in bytes");
  desc.add_options()("max_ralt_size",
                     po::value<double>(&arg_max_ralt_size)->required(),
                     "Max physical size of ralt in bytes");
  desc.add_options()("ralt_path",
                     po::value<std::string>(&ralt_path_str)->required(),
                     "Path to RALT");
  desc.add_options()("compaction_pri,p",
                     po::value<int>(&compaction_pri)->required(),
                     "Method to pick SST to compact (rocksdb::CompactionPri)");

  // Options for RALT
  desc.add_options()("enable_auto_tuning", "enable auto-tuning");
  desc.add_options()("ralt_bloom_bits", po::value<>(&ralt_options.bloom_bits),
                     "The number of bits per key in RALT bloom filter.");
  desc.add_options()("ralt_exp_smoothing_factor",
                     po::value<>(&ralt_options.exp_smoothing_factor));

  po::variables_map vm;
  po::store(po::parse_command_line(argc, argv, desc), vm);
  if (vm.count("help")) {
    std::cerr << desc << std::endl;
    return 1;
  }
  po::notify(vm);

  uint64_t hot_set_size_limit = arg_max_hot_set_size;
  uint64_t max_ralt_size = arg_max_ralt_size;

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
    switches = 0xf;
  } else {
    std::istringstream in(std::move(arg_switches));
    in >> std::hex >> switches;
  }

  std::filesystem::path db_path(arg_db_path);
  std::filesystem::path ralt_path(ralt_path_str);
  options.db_paths = decode_db_paths(arg_db_paths);
  options.compaction_pri = static_cast<rocksdb::CompactionPri>(compaction_pri);
  options.statistics = rocksdb::CreateDBStatistics();
  options.compression = rocksdb::CompressionType::kNoCompression;
  // Doesn't make sense for tiered storage
  options.level_compaction_dynamic_level_bytes = false;
  // The ttl feature will try to compact old data into the last level, which is
  // not compatible with the retention of HotRAP. So we disable the ttl feature.
  options.ttl = 0;

  table_options.block_cache = rocksdb::NewLRUCache(cache_size);
  table_options.filter_policy.reset(rocksdb::NewBloomFilterPolicy(10, false));
  options.table_factory.reset(
      rocksdb::NewBlockBasedTableFactory(table_options));

  if (vm.count("optimize_filters_for_hits")) {
    options.optimize_filters_for_hits = true;
  }

  if (load_phase_rate_limit) {
    rocksdb::RateLimiter *rate_limiter =
        rocksdb::NewGenericRateLimiter(load_phase_rate_limit, 100 * 1000, 10,
                                       rocksdb::RateLimiter::Mode::kAllIo);
    options.rate_limiter.reset(rate_limiter);
    work_options.rate_limiter = options.rate_limiter;
  }

  if (work_options.load) {
    std::cerr << "Emptying directories\n";
    empty_directory(db_path);
    for (auto path : options.db_paths) {
      empty_directory(path.path);
    }
    std::cerr << "Creating database\n";
    options.create_if_missing = true;
  }

  size_t first_level_in_last_tier = calc_first_level_in_sd(options);
  calc_fd_size_ratio(options, first_level_in_last_tier, max_ralt_size);

  auto level_size_path_id = predict_level_assignment(options);
  rusty_assert_eq(level_size_path_id.size(), first_level_in_last_tier + 1);
  for (size_t level = 0; level < level_size_path_id.size() - 1; ++level) {
    auto p = level_size_path_id[level].second;
    std::cerr << level << ' ' << options.db_paths[p].path << ' '
              << level_size_path_id[level].first << std::endl;
  }
  auto p = level_size_path_id[first_level_in_last_tier].second;
  std::cerr << level_size_path_id.size() - 1 << "+ " << options.db_paths[p].path
            << ' ' << level_size_path_id[first_level_in_last_tier].first
            << std::endl;
  auto first_level_in_last_tier_path = db_path / "first-level-in-last-tier";
  if (std::filesystem::exists(first_level_in_last_tier_path)) {
    std::ifstream first_level_in_last_tier_in(first_level_in_last_tier_path);
    rusty_assert(first_level_in_last_tier_in);
    std::string first_level_in_last_tier_stored;
    std::getline(first_level_in_last_tier_in, first_level_in_last_tier_stored);
    rusty_assert_eq((size_t)std::atoi(first_level_in_last_tier_stored.c_str()),
                    first_level_in_last_tier);
  } else {
    std::ofstream(first_level_in_last_tier_path)
        << first_level_in_last_tier << std::endl;
  }

  std::shared_ptr<RaltWrapper> ralt = nullptr;
  if (first_level_in_last_tier != 0) {
    uint64_t fd_size = options.db_paths[0].target_size;
    ralt = std::make_shared<RaltWrapper>(
        ralt_options, options.comparator, ralt_path_str,
        first_level_in_last_tier - 1, hot_set_size_limit, max_ralt_size,
        switches, hot_set_size_limit, hot_set_size_limit, fd_size);
    options.ralt = ralt;
  }

  rocksdb::DB *db;
  auto s = rocksdb::DB::Open(options, db_path.string(), &db);
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
  work_options.options = &options;
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
  work_options.enable_fast_generator = vm.count("enable_fast_generator");
  std::string workload = vm["workload"].as<std::string>();
  if (workload == "file") {
    work_options.workload_type = WorkloadType::ConfigFile;
  } else if (workload == "u24685531") {
    work_options.workload_type = WorkloadType::u24685531;
  } else if (workload == "2-4-6-8") {
    work_options.workload_type = WorkloadType::hotspot_2_4_6_8;
  } else if (workload == "5-shift-5") {
    work_options.workload_type = WorkloadType::hotspot_5_shift_5;
  } else {
    rusty_panic("Unknown workload %s", workload.c_str());
  }
  if (work_options.enable_fast_generator) {
    if (work_options.workload_type == WorkloadType::ConfigFile) {
      std::string workload_file = vm["workload_file"].as<std::string>();
      work_options.ycsb_gen_options =
          YCSBGen::YCSBGeneratorOptions::ReadFromFile(workload_file);
    }
    work_options.export_key_only_trace = vm.count("export_key_only_trace");
  } else {
    rusty_assert(vm.count("workload_file") == 0,
                 "workload_file only works with built-in generator!");
    rusty_assert(vm.count("export_key_only_trace") == 0,
                 "export_key_only_trace only works with built-in generator!");
    work_options.ycsb_gen_options = YCSBGen::YCSBGeneratorOptions();
  }
  work_options.export_ans_xxh64 = vm.count("export_ans_xxh64");

  work_options.block_cache = table_options.block_cache;

  AutoTuner *autotuner = nullptr;
  if (vm.count("enable_auto_tuning") && ralt) {
    assert(first_level_in_sd > 0);
    uint64_t fd_size = options.db_paths[0].target_size;
    autotuner = new AutoTuner(*db, first_level_in_last_tier, fd_size / 20, 0.85,
                              fd_size / 20);
  }

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

  /* Statistics of RALT */
  if (ralt) {
    if (tester.work_options().switches & MASK_COUNT_ACCESS_HOT_PER_TIER) {
      auto counters = ralt->hit_hot_count();
      assert(counters.size() == 2);
      std::cerr << "Access hot per tier: " << counters[0] << ' ' << counters[1]
                << "\nAccess FD hot: " << ralt->count_access_fd_hot()
                << "\nAccess FD cold: " << ralt->count_access_fd_cold() << '\n';
    }
  }

  should_stop.store(true, std::memory_order_relaxed);
  stat_printer.join();
  period_print_thread.join();
  delete autotuner;
  delete db;

  return 0;
}
