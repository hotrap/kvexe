#include <sys/resource.h>

#include <counter_timer_vec.hpp>

#include "rocksdb/compaction_router.h"
#include "test.hpp"
#include "viscnts.h"

typedef uint16_t field_size_t;

thread_local std::optional<std::ofstream> key_hit_level_out;
std::optional<std::ofstream> &get_key_hit_level_out() {
  return key_hit_level_out;
}

std::vector<rocksdb::DbPath> decode_db_paths(std::string db_paths) {
  std::istringstream in(db_paths);
  std::vector<rocksdb::DbPath> ret;
  rusty_assert_eq(in.get(), '{', "Invalid db_paths");
  char c = static_cast<char>(in.get());
  if (c == '}') return ret;
  rusty_assert_eq(c, '{', "Invalid db_paths");
  while (1) {
    std::string path;
    size_t size;
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

// Return the first level in CD
size_t calculate_multiplier_addtional(rocksdb::Options &options,
                                      size_t max_hot_set_size) {
  rusty_assert_eq(options.db_paths.size(), 2.0);
  size_t sd_size = options.db_paths[0].target_size;
  for (double x : options.max_bytes_for_level_multiplier_additional) {
    rusty_assert(x - 1 < 1e-6);
  }
  options.max_bytes_for_level_multiplier_additional.clear();
  size_t level = 0;
  uint64_t level_size = options.max_bytes_for_level_base;
  while (level_size <= sd_size) {
    sd_size -= level_size;
    if (level > 0) {
      level_size *= options.max_bytes_for_level_multiplier;
    }
    level += 1;
  }
  level_size /= options.max_bytes_for_level_multiplier;
  // It seems that L0 and L1 are not affected by
  // options.max_bytes_for_level_multiplier_additional
  if (level <= 2) return level;
  size_t last_level_in_sd = level - 1;
  for (size_t i = 1; i < last_level_in_sd; ++i) {
    options.max_bytes_for_level_multiplier_additional.push_back(1.0);
  }
  // Multiply 0.99 to make room for floating point error
  options.max_bytes_for_level_multiplier_additional.push_back(
      1 + (double)sd_size / level_size * 0.99);
  uint64_t last_level_in_sd_size = sd_size + level_size;
  rusty_assert(last_level_in_sd_size > max_hot_set_size);
  uint64_t last_level_in_sd_effective_size =
      last_level_in_sd_size - max_hot_set_size;
  uint64_t first_level_in_cd_size = last_level_in_sd_effective_size * 10;
  options.max_bytes_for_level_multiplier_additional.push_back(
      (double)first_level_in_cd_size / (last_level_in_sd_size * 10));
  return level;
}

double MaxBytesMultiplerAdditional(const rocksdb::Options &options, int level) {
  if (level >= static_cast<int>(
                   options.max_bytes_for_level_multiplier_additional.size())) {
    return 1;
  }
  return options.max_bytes_for_level_multiplier_additional[level];
}

std::vector<std::pair<size_t, std::string>> predict_level_assignment(
    const rocksdb::Options &options) {
  std::vector<std::pair<size_t, std::string>> ret;
  uint32_t p = 0;
  int level = 0;
  assert(!options.db_paths.empty());

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
      rusty_assert_eq(ret.size(), (size_t)level);
      ret.emplace_back(level_size, options.db_paths[p].path);
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
  rusty_assert_eq(ret.size(), (size_t)level);
  ret.emplace_back(level_size, options.db_paths[p].path);
  return ret;
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

enum class PerLevelTimerType : size_t {
  kHitLevel = 0,
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

template <typename T>
class TimedIter : public rocksdb::TraitIterator<T> {
 public:
  TimedIter(std::unique_ptr<rocksdb::TraitIterator<T>> iter)
      : iter_(std::move(iter)) {}
  rocksdb::optional<T> next() override {
    auto guard = timers.timer(TimerType::kNextHot).start();
    return iter_->next();
  }

 private:
  std::unique_ptr<rocksdb::TraitIterator<T>> iter_;
};

class RouterVisCnts : public rocksdb::CompactionRouter {
 public:
  RouterVisCnts(const rocksdb::Comparator *ucmp, std::filesystem::path dir,
                int tier0_last_level, size_t init_hot_set_size,
                size_t max_viscnts_size, uint64_t switches, 
                size_t max_hot_set_size, size_t min_hot_set_size)
      : switches_(switches),
        vc_(VisCnts::New(ucmp, dir.c_str(), init_hot_set_size,
                         max_hot_set_size, min_hot_set_size, max_viscnts_size)),
        tier0_last_level_(tier0_last_level),
        count_access_hot_per_tier_{0, 0} {}
  const char *Name() const override { return "RouterVisCnts"; }
  size_t Tier(int level) override {
    if (level <= tier0_last_level_) {
      return 0;
    } else {
      return 1;
    }
  }
  void HitLevel(int level, rocksdb::Slice key) override {
    auto guard =
        per_level_timers.timer(level, PerLevelTimerType::kHitLevel).start();

    size_t tier = Tier(level);

    if (switches_ & MASK_COUNT_ACCESS_HOT_PER_TIER) {
      if (vc_.IsHot(key)) count_access_hot_per_tier_[tier].fetch_add(1);
    }

    count_access_per_tier_[tier].fetch_add(1, std::memory_order_relaxed);

    if (get_key_hit_level_out().has_value()) {
      get_key_hit_level_out().value() << key.ToString() << ' ' << level << '\n';
    }
  }
  void Access(rocksdb::Slice key, size_t vlen) override {
    auto guard = timers.timer(TimerType::kAccess).start();
    vc_.Access(key, vlen);
  }

  bool IsStablyHot(rocksdb::Slice key) override {
    auto guard = timers.timer(TimerType::kIsStablyHot).start();
    return vc_.IsStablyHot(key);
  }
  // The returned pointer will stay valid until the next call to Seek or
  // NextHot with this iterator
  rocksdb::CompactionRouter::Iter LowerBound(rocksdb::Slice key) override {
    auto guard = timers.timer(TimerType::kLowerBound).start();
    return rocksdb::CompactionRouter::Iter(
        std::make_unique<TimedIter<rocksdb::HotRecInfo>>(vc_.LowerBound(key)));
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

  bool get_viscnts_int_property(std::string_view property, uint64_t *value) {
    return vc_.GetIntProperty(property, value);
  }

  std::vector<size_t> hit_hot_count() {
    std::vector<size_t> ret;
    for (size_t i = 0; i < 2; ++i)
      ret.push_back(
          count_access_hot_per_tier_[i].load(std::memory_order_relaxed));
    return ret;
  }

  std::vector<size_t> hit_tier_count() {
    std::vector<size_t> ret;
    for (size_t i = 0; i < 2; ++i)
      ret.push_back(count_access_per_tier_[i].load(std::memory_order_relaxed));
    return ret;
  }

  VisCnts &get_vc() { return vc_; }

 private:
  const uint64_t switches_;
  VisCnts vc_;
  int tier0_last_level_;

  std::atomic<size_t> count_access_hot_per_tier_[2];
  std::atomic<size_t> count_access_per_tier_[2];
};

class HitRateMonitor {
 public:
  HitRateMonitor(size_t threshold) : threshold_(threshold) {}

  void BeginPeriod(const std::vector<size_t> &hr) {
    lst_hr_ = hr;
    period_first_hr_ = hr;
    is_stable_ = false;
    tick_ = 0;
    max_rate_ = 0;
    min_rate_ = 1;
    eq_tick_ = 0;
    is_in_per_ = true;
  }


  double AddPeriodData(const std::vector<size_t>& hr) {
    if ((ssize_t)hr[1] + hr[0] - lst_hr_[1] - lst_hr_[0] < threshold_) {
      return -1;
    }
    double rate = CalcRate(lst_hr_, hr);
    if (max_rate_ + 0.005 < rate || min_rate_ - 0.005 > rate) {
      max_rate_ = std::max(max_rate_, rate);
      min_rate_ = std::min(min_rate_, rate);
      eq_tick_ = 0;
    } else {
      eq_tick_ += 1;
    }
    if (eq_tick_ == 1) {
      period_first_hr_ = hr;
    }
    if (eq_tick_ == 4) {
      is_stable_ = true;
    }
    lst_hr_ = hr;
    tick_ += 1;
    return rate;
  }

  bool IsStable() const { return is_stable_; }

  double GetStableRate() const { return CalcRate(period_first_hr_, lst_hr_); }

  bool IsInPeriod() const { return is_in_per_; }

  void EndPeriod() { is_in_per_ = false; }

 private:
  double CalcRate(const std::vector<size_t> &L,
                  const std::vector<size_t> &R) const {
    double rate = ((double)R[0] - L[0]) / ((double)R[0] - L[0] + R[1] - L[1]);
    return rate;
  }

  double lst_rate_{0};
  double max_rate_{0};
  double min_rate_{0};
  bool is_stable_{false};
  bool is_in_per_{false};
  size_t tick_{0};
  size_t eq_tick_{0};
  size_t threshold_{100};
  std::vector<size_t> lst_hr_;
  std::vector<size_t> period_first_hr_;
};

class VisCntsUpdater {
 public:
  VisCntsUpdater(const std::filesystem::path &db_path, size_t hot_set_size,
                 size_t phy_size, size_t max_vc_hot_set_size,
                 size_t min_vc_hot_set_size, size_t wait_op,
                 size_t wait_time_ns, RouterVisCnts &router)
      : cur_vc_hot_set_size_(hot_set_size),
        cur_vc_phy_size_(phy_size),
        max_vc_hot_set_size_(max_vc_hot_set_size),
        min_vc_hot_set_size_(min_vc_hot_set_size),
        wait_op_(wait_op),
        wait_time_ns_(wait_time_ns),
        router_(router),
        log_(db_path / "vc_log"),
        hr_mon_(wait_op) {
      th_ = std::thread([&](){
        update_thread();
      });
    }

  ~VisCntsUpdater() { Stop(); }

  void Stop() {
    stop_signal_ = true;
    th_.join();
  }

  size_t GetCurHotSetSizeLimit() const {
    return router_.get_vc().GetHotSetSizeLimit();
  }

  size_t GetCurPhySizeLimit() const {
    return router_.get_vc().GetPhySizeLimit();
  }

  private:
    void update_thread() {
      while (!stop_signal_) {
        std::this_thread::sleep_for(std::chrono::nanoseconds(wait_time_ns_));
        if (stop_signal_) {
          break;
        }
        auto hits = router_.hit_tier_count();
        if (!hr_mon_.IsInPeriod()) {
          hr_mon_.BeginPeriod(hits);
        } else {
          double rate = hr_mon_.AddPeriodData(hits);
          log_ << "[VC Updater] " << timestamp_ns() << " Rate: " << rate << std::endl;
        }
        double hs_step = (max_vc_hot_set_size_ - min_vc_hot_set_size_) / 20.0;
        ssize_t new_vc_hs = cur_vc_hot_set_size_;
        ssize_t new_vc_phy = cur_vc_phy_size_;
        if (hr_mon_.IsStable()) {
          double rate = hr_mon_.GetStableRate();
          log_ << "[VC Updater] Stable Rate: " << rate << std::endl;
          if (lst_hit_rate_ == -1) {
            // Get the stable hit rate
            lst_hit_rate_ = rate;
            lst_choose_ = -1;
            cur_vc_hot_set_size_ = new_vc_hs = max_vc_hot_set_size_;
            phase_num_ = 0;
          }
          if (lst_hit_rate_ - rate > 0.1) {
            // The data distribution may be changed.
            lst_hit_rate_ = rate;
            cur_vc_hot_set_size_ = new_vc_hs = max_vc_hot_set_size_;
            phase_num_ = 0;
          }

          // maximum hit rate updates,
          if (lst_hit_rate_ < rate - 0.01) {
            lst_hit_rate_ = rate;
            lst_choose_ = std::max(-1., lst_choose_ * 2);
            new_vc_hs -= lst_choose_ * hs_step;
          }

          if (phase_num_ == 0) {
            double real_hs = router_.get_vc().GetRealHotSetSize();
            double real_ps = router_.get_vc().GetRealPhySize();
            log_ << "[VC Updater] Real HS: " << real_hs
                << ", Real PS: " << real_ps << std::endl;
            if (real_hs > cur_vc_hot_set_size_ * 0.95) {
              phase_num_ = 1;
              lst_choose_ = -1;
              router_.get_vc().SetProperPhysicalSizeLimit();
              cur_vc_phy_size_ = router_.get_vc().GetPhySizeLimit();
            } else if (router_.get_vc().DecayCount() > 0 && real_hs > 0) {
              double delta =
                  (cur_vc_hot_set_size_ - real_hs) / (double)real_hs * real_ps;
              delta = std::min(delta, (double)(128 << 20));
              new_vc_phy = real_ps + delta;
            }
          } else if (phase_num_ == 1) {
            log_ << "[VC Updater] Lst Choose: " << lst_choose_
                << ", Lst Ret: " << lst_ret_cur_hot_set_size_ << std::endl;
            if (router_.get_vc().DecayCount() > 0) {
              // Try to decrease hot set size.
              if (lst_hit_rate_ < rate + 0.01) {
                if (lst_ret_cur_hot_set_size_ >=
                    new_vc_hs + lst_choose_ * hs_step) {
                  lst_choose_ = std::min(-0.02, lst_choose_ * 0.5);
                }
                new_vc_hs += lst_choose_ * hs_step;
              } else {
                lst_ret_cur_hot_set_size_ = new_vc_hs;
                new_vc_hs -= lst_choose_ * hs_step;
              }
            }
            if (lst_choose_ == -0.02) {
              phase_num_ = 2;
              router_.get_vc().SetProperPhysicalSizeLimit();
              cur_vc_phy_size_ = router_.get_vc().GetPhySizeLimit();
            }
          }

          hr_mon_.EndPeriod();
          hr_mon_.BeginPeriod(hits);
        }

        new_vc_hs = std::min(std::max(min_vc_hot_set_size_, new_vc_hs),
                            max_vc_hot_set_size_);
        if (new_vc_hs != cur_vc_hot_set_size_ || new_vc_phy != cur_vc_phy_size_) {
          cur_vc_hot_set_size_ = new_vc_hs;
          cur_vc_phy_size_ = new_vc_phy;
          router_.get_vc().SetAllSizeLimit(cur_vc_hot_set_size_,
                                          cur_vc_phy_size_);
        }
      }
    }

  ssize_t cur_vc_hot_set_size_;
  ssize_t cur_vc_phy_size_;
  ssize_t max_vc_hot_set_size_;
  ssize_t min_vc_hot_set_size_;
  ssize_t wait_op_;
  ssize_t wait_time_ns_;
  ssize_t lst_progress_{0};
  ssize_t lst_time_{0};
  size_t phase_num_{0};
  double step_rate_{1};
  RouterVisCnts &router_;
  std::ofstream log_;

  bool stop_signal_{false};
  std::thread th_;
  std::condition_variable cv_;
  std::mutex m_;

  std::atomic<size_t> progress_{0};
  HitRateMonitor hr_mon_;
  double lst_hit_rate_{-1};
  double lst_choose_{-1};
  ssize_t lst_ret_cur_hot_set_size_{0};
  ssize_t resize_tick_{0};
};


class VisCntsUpdater2 {
 public:
  VisCntsUpdater2(const std::filesystem::path &db_path, size_t max_vc_hot_set_size,
                 size_t min_vc_hot_set_size, size_t wait_op,
                 size_t wait_time_ns, RouterVisCnts &router)
      : wait_op_(wait_op),
        wait_time_ns_(wait_time_ns),
        max_vc_hot_set_size_(max_vc_hot_set_size),
        min_vc_hot_set_size_(min_vc_hot_set_size),
        router_(router),
        log_(db_path / "vc_log") {
      th_ = std::thread([&](){
        update_thread();
      });
    }

  ~VisCntsUpdater2() { Stop(); }

  void Stop() {
    stop_signal_ = true;
    th_.join();
  }

  size_t GetCurHotSetSizeLimit() const {
    return router_.get_vc().GetHotSetSizeLimit();
  }

  size_t GetCurPhySizeLimit() const {
    return router_.get_vc().GetPhySizeLimit();
  }

  private:
    void update_thread() {
      while (!stop_signal_) {
        std::this_thread::sleep_for(std::chrono::nanoseconds(wait_time_ns_));
        if (stop_signal_) {
          break;
        }
        double hs_step = max_vc_hot_set_size_ / 20.0;
        if (router_.get_vc().DecayCount() > 3) {
          auto rate = router_.get_vc().GetRealPhySize() / (double) router_.get_vc().GetRealHotSetSize();
          auto delta = rate * hs_step; //std::max<size_t>(rate * hs_step, (64 << 20));
          auto phy_size = router_.get_vc().GetRealPhySize() + delta;
          std::cerr<<"rate " <<rate<<"," <<phy_size<<std::endl;
          router_.get_vc().SetPhysicalSizeLimit(phy_size);
        }
      }
    }

  ssize_t wait_op_;
  ssize_t wait_time_ns_;
  ssize_t max_vc_hot_set_size_;
  ssize_t min_vc_hot_set_size_;
  RouterVisCnts &router_;
  std::ofstream log_;

  bool stop_signal_{false};
  std::thread th_;
};

void print_vc_param(RouterVisCnts& router, WorkOptions *work_options,
               std::atomic<bool> *should_stop) {
  const std::filesystem::path &db_path = work_options->db_path;
  auto vc_parameter_path = db_path / "vc_param";
  std::ofstream out(vc_parameter_path);
  while (!should_stop->load(std::memory_order_relaxed)) {
    std::this_thread::sleep_for(std::chrono::seconds(1));
    auto timestamp = timestamp_ns();
    out << timestamp << " " << router.get_vc().GetHotSetSizeLimit() << " "
        << router.get_vc().GetPhySizeLimit() << std::endl;
  }
}

void bg_stat_printer(WorkOptions *work_options, const rocksdb::Options *options,
                     std::atomic<bool> *should_stop) {
  rocksdb::DB *db = work_options->db;
  auto router = static_cast<RouterVisCnts *>(options->compaction_router);
  const std::filesystem::path &db_path = work_options->db_path;

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
                "get-cpu-nanos "
             << VisCnts::Properties::kCompactionThreadCPUNanos << ' '
             << VisCnts::Properties::kFlushThreadCPUNanos << ' '
             << VisCnts::Properties::kDecayThreadCPUNanos << ' '
             << "viscnts.compaction.cpu.nanos viscnts.flush.cpu.nanos "
                "viscnts.decay.scan.cpu.nanos viscnts.decay.write.cpu.nanos\n";

  std::ofstream promoted_or_retained_out(db_path /
                                         "promoted-or-retained-bytes");
  promoted_or_retained_out
      << "Timestamp(ns) by-flush 2sdlast 2cdfront retained\n";

  std::ofstream not_promoted_bytes_out(db_path / "not-promoted-bytes");
  not_promoted_bytes_out << "Timestamp(ns) not-stably-hot has-newer-version\n";

  std::ofstream num_accesses_out(db_path / "num-accesses");

  std::ofstream viscnts_io_out(db_path / "viscnts-io");
  viscnts_io_out << "Timestamp(ns) read write\n";

  auto stats = options->statistics;

  std::ofstream rand_read_bytes_out(db_path / "rand-read-bytes");

  auto interval = rusty::time::Duration::from_secs(1);
  auto next_begin = rusty::time::Instant::now() + interval;
  while (!should_stop->load(std::memory_order_relaxed)) {
    auto timestamp = timestamp_ns();

    progress_out << timestamp << ' '
                 << work_options->progress->load(std::memory_order_relaxed)
                 << ' '
                 << work_options->progress_get->load(std::memory_order_relaxed)
                 << std::endl;

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
    uint64_t viscnts_compaction_thread_cpu_nanos;
    rusty_assert(router->get_viscnts_int_property(
        VisCnts::Properties::kCompactionThreadCPUNanos,
        &viscnts_compaction_thread_cpu_nanos));
    uint64_t viscnts_flush_thread_cpu_nanos;
    rusty_assert(router->get_viscnts_int_property(
        VisCnts::Properties::kFlushThreadCPUNanos,
        &viscnts_flush_thread_cpu_nanos));
    uint64_t viscnts_decay_thread_cpu_nanos;
    rusty_assert(router->get_viscnts_int_property(
        VisCnts::Properties::kDecayThreadCPUNanos,
        &viscnts_decay_thread_cpu_nanos));
    uint64_t viscnts_compaction_cpu_nanos;
    rusty_assert(router->get_viscnts_int_property(
        VisCnts::Properties::kCompactionCPUNanos,
        &viscnts_compaction_cpu_nanos));
    uint64_t viscnts_flush_cpu_nanos;
    rusty_assert(router->get_viscnts_int_property(
        VisCnts::Properties::kFlushCPUNanos, &viscnts_flush_cpu_nanos));
    uint64_t viscnts_decay_scan_cpu_nanos;
    rusty_assert(router->get_viscnts_int_property(
        VisCnts::Properties::kDecayScanCPUNanos,
        &viscnts_decay_scan_cpu_nanos));
    uint64_t viscnts_decay_write_cpu_nanos;
    rusty_assert(router->get_viscnts_int_property(
        VisCnts::Properties::kDecayWriteCPUNanos,
        &viscnts_decay_write_cpu_nanos));
    timers_out << timestamp << ' ' << compaction_cpu_micros << ' '
               << put_cpu_nanos.load(std::memory_order_relaxed) << ' '
               << get_cpu_nanos.load(std::memory_order_relaxed) << ' '
               << viscnts_compaction_thread_cpu_nanos << ' '
               << viscnts_flush_thread_cpu_nanos << ' '
               << viscnts_decay_thread_cpu_nanos << ' '
               << viscnts_compaction_cpu_nanos << ' ' << viscnts_flush_cpu_nanos
               << ' ' << viscnts_decay_scan_cpu_nanos << ' '
               << viscnts_decay_write_cpu_nanos << std::endl;

    std::string rand_read_bytes;
    rusty_assert(db->GetProperty(rocksdb::DB::Properties::kRandReadBytes,
                                 &rand_read_bytes));
    rand_read_bytes_out << timestamp << ' ' << rand_read_bytes << std::endl;

    promoted_or_retained_out
        << timestamp << ' '
        << stats->getTickerCount(rocksdb::PROMOTED_FLUSH_BYTES) << ' '
        << stats->getTickerCount(rocksdb::PROMOTED_2SDLAST_BYTES) << ' '
        << stats->getTickerCount(rocksdb::PROMOTED_2CDFRONT_BYTES) << ' '
        << stats->getTickerCount(rocksdb::RETAINED_BYTES) << std::endl;

    not_promoted_bytes_out
        << timestamp << ' '
        << stats->getTickerCount(rocksdb::NOT_STABLY_HOT_BYTES) << ' '
        << stats->getTickerCount(rocksdb::HAS_NEWER_VERSION_BYTES) << std::endl;

    size_t num_level = per_level_timers.len();
    num_accesses_out << timestamp;
    for (size_t level = 0; level < num_level; ++level) {
      num_accesses_out
          << ' '
          << per_level_timers.timer(level, PerLevelTimerType::kHitLevel).count();
    }
    num_accesses_out << std::endl;

    uint64_t viscnts_read;
    rusty_assert(router->get_viscnts_int_property(
        VisCnts::Properties::kReadBytes, &viscnts_read));
    uint64_t viscnts_write;
    rusty_assert(router->get_viscnts_int_property(
        VisCnts::Properties::kWriteBytes, &viscnts_write));
    viscnts_io_out << timestamp << ' ' << viscnts_read << ' ' << viscnts_write
                   << std::endl;

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

  rocksdb::Options options;
  WorkOptions work_option;

  namespace po = boost::program_options;
  po::options_description desc("Available options");
  std::string format;
  std::string arg_switches;
  size_t num_threads;

  std::string arg_db_path;
  std::string arg_db_paths;
  std::string viscnts_path_str;
  size_t cache_size;

  double arg_max_hot_set_size;
  double arg_max_viscnts_size;
  int compaction_pri;

  // Options of executor
  desc.add_options()("help", "Print help message");
  desc.add_options()("format,f",
                     po::value<std::string>(&format)->default_value("ycsb"),
                     "Trace format: plain/ycsb");
  desc.add_options()(
      "load", "Execute the load phase. Will empty the directories first.");
  desc.add_options()(
      "run",
      "Execute the run phase. "
      "If --load is not provided, the run phase will be directly executed "
      "without executing the load phase, and the directories won't be cleaned "
      "up. "
      "If none of --load and --run is provided, the both phases will be "
      "executed.");
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
  desc.add_options()("enable_fast_process",
                     "Enable fast process including ignoring kNotFound and "
                     "pushing operations in one channel.");
  desc.add_options()("enable_fast_generator", "Enable fast generator");
  desc.add_options()("workload_file", po::value<std::string>(),
                     "Workload file used in built-in generator");
  desc.add_options()("export_key_only_trace",
                     "Export key-only trace generated by built-in generator.");

  // Options of rocksdb
  desc.add_options()("max_background_jobs", po::value<int>(), "");
  desc.add_options()("level0_file_num_compaction_trigger", po::value<int>(),
                     "Number of files in level-0 when compactions start");
  desc.add_options()(
      "use_direct_reads",
      po::value<bool>(&options.use_direct_reads)->default_value(true), "");
  desc.add_options()(
      "use_direct_io_for_flush_and_compaction",
      po::value<bool>(&options.use_direct_io_for_flush_and_compaction)
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
  desc.add_options()("block_size", po::value<size_t>(), "Default: 4096");
  desc.add_options()("max_bytes_for_level_base", po::value<uint64_t>(), "");
  desc.add_options()("optimize_filters_for_hits",
                     "Do not build filters for the last level");

  // Options for hotrap
  desc.add_options()("max_hot_set_size",
                     po::value<double>(&arg_max_hot_set_size)->required(),
                     "Max hot set size in bytes");
  desc.add_options()("max_viscnts_size",
                     po::value<double>(&arg_max_viscnts_size)->required(),
                     "Max physical size of viscnts in bytes");
  desc.add_options()("viscnts_path",
                     po::value<std::string>(&viscnts_path_str)->required(),
                     "Path to VisCnts");
  desc.add_options()("compaction_pri,p",
                     po::value<int>(&compaction_pri)->required(),
                     "Method to pick SST to compact (rocksdb::CompactionPri)");
  desc.add_options()(
      "db_paths_soft_size_limit_multiplier",
      po::value<double>(&work_option.db_paths_soft_size_limit_multiplier)
          ->default_value(1.1));

  desc.add_options()("enable_dynamic_vc_param", "enable_dynamic_vc_param");

  desc.add_options()("enable_dynamic_vc_param_in_lsm", "enable_dynami_vc_param_in_lsm");

  desc.add_options()("enable_dynamic_only_vc_phy_size", "enable_dynamic_only_vc_phy_size");

  po::variables_map vm;
  po::store(po::parse_command_line(argc, argv, desc), vm);
  if (vm.count("help")) {
    std::cerr << desc << std::endl;
    return 1;
  }
  po::notify(vm);

  size_t max_hot_set_size = arg_max_hot_set_size;
  size_t max_viscnts_size = arg_max_viscnts_size;

  if (vm.count("load")) {
    work_option.load = true;
  }
  if (vm.count("run")) {
    work_option.run = true;
  }
  if (work_option.load == false && work_option.run == false) {
    work_option.load = true;
    work_option.run = true;
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
  std::filesystem::path viscnts_path(viscnts_path_str);
  options.db_paths = decode_db_paths(arg_db_paths);
  options.compaction_pri = static_cast<rocksdb::CompactionPri>(compaction_pri);
  options.statistics = rocksdb::CreateDBStatistics();
  options.compression = rocksdb::CompressionType::kNoCompression;
  // Doesn't make sense for tiered storage
  options.level_compaction_dynamic_level_bytes = false;

  rocksdb::BlockBasedTableOptions table_options;
  table_options.block_cache = rocksdb::NewLRUCache(cache_size);
  table_options.filter_policy.reset(rocksdb::NewBloomFilterPolicy(10, false));
  if (vm.count("block_size")) {
    table_options.block_size = vm["block_size"].as<size_t>();
  }
  options.table_factory.reset(
      rocksdb::NewBlockBasedTableFactory(table_options));

  if (vm.count("max_background_jobs")) {
    options.max_background_jobs = vm["max_background_jobs"].as<int>();
  }

  if (vm.count("level0_file_num_compaction_trigger")) {
    options.level0_file_num_compaction_trigger =
        vm["level0_file_num_compaction_trigger"].as<int>();
  }
  if (vm.count("max_bytes_for_level_base")) {
    options.max_bytes_for_level_base =
        vm["max_bytes_for_level_base"].as<uint64_t>();
  }
  if (vm.count("optimize_filters_for_hits")) {
    options.optimize_filters_for_hits = true;
  }

  size_t first_level_in_cd =
      calculate_multiplier_addtional(options, max_hot_set_size);
  std::cerr << "options.max_bytes_for_level_multiplier_additional: ";
  print_vector(options.max_bytes_for_level_multiplier_additional);
  std::cerr << std::endl;
  auto ret = predict_level_assignment(options);
  rusty_assert_eq(ret.size() - 1, first_level_in_cd);
  for (size_t level = 0; level < first_level_in_cd; ++level) {
    std::cerr << level << ' ' << ret[level].second << ' ' << ret[level].first
              << std::endl;
  }
  std::cerr << first_level_in_cd << "+ " << ret[first_level_in_cd].second << ' '
            << ret[first_level_in_cd].first << std::endl;
  if (options.db_paths.size() == 1) {
    first_level_in_cd = 100;
  }

  RouterVisCnts *router = nullptr;
  VisCntsUpdater2 *updater = nullptr;
  if (first_level_in_cd != 0) {
    if (vm.count("enable_dynamic_vc_param_in_lsm")) {
      router = new RouterVisCnts(options.comparator, viscnts_path_str,
                              first_level_in_cd - 1, max_hot_set_size,
                              max_viscnts_size, switches, options.db_paths[0].target_size * 0.7, 
                              options.db_paths[0].target_size * 0.05);
    } else {
      router = new RouterVisCnts(options.comparator, viscnts_path_str,
                                first_level_in_cd - 1, max_hot_set_size,
                                max_viscnts_size, switches, max_hot_set_size, max_hot_set_size);  
    }
    
    options.compaction_router = router;
  }

  rocksdb::DB *db;
  if (work_option.load) {
    std::cerr << "Emptying directories\n";
    empty_directory(db_path);
    for (auto path : options.db_paths) {
      empty_directory(path.path);
    }
    std::ofstream(db_path / "first-level-in-cd")
        << first_level_in_cd << std::endl;

    std::cerr << "Creating database\n";
    options.create_if_missing = true;
  }

  // if (vm.count("enable_dynamic_vc_param") && router) {
  //   updater = new VisCntsUpdater(db_path, max_hot_set_size, max_viscnts_size, options.db_paths[0].target_size * 0.7, options.db_paths[0].target_size * 0.1, 5e5, 2e9, *router);
  // }

  if (vm.count("enable_dynamic_only_vc_phy_size") && router) {
    updater = new VisCntsUpdater2(db_path,options.db_paths[0].target_size * 0.7, options.db_paths[0].target_size * 0.1, 5e5, 20e9, *router);
  }

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

  std::atomic<size_t> progress(0);
  std::atomic<size_t> progress_get(0);

  work_option.db = db;
  work_option.switches = switches;
  work_option.db_path = db_path;
  work_option.progress = &progress;
  work_option.progress_get = &progress_get;
  work_option.num_threads = num_threads;
  work_option.enable_fast_process = vm.count("enable_fast_process");
  work_option.format_type =
      format == "ycsb" ? FormatType::YCSB : FormatType::Plain;
  work_option.enable_fast_generator = vm.count("enable_fast_generator");
  if (work_option.enable_fast_generator) {
    std::string workload_file = vm["workload_file"].as<std::string>();
    work_option.ycsb_gen_options =
        YCSBGen::YCSBGeneratorOptions::ReadFromFile(workload_file);
    work_option.export_key_only_trace = vm.count("export_key_only_trace");
  } else {
    rusty_assert(vm.count("workload_file") == 0,
                 "workload_file only works with built-in generator!");
    rusty_assert(vm.count("export_key_only_trace") == 0,
                 "export_key_only_trace only works with built-in generator!");
    work_option.ycsb_gen_options = YCSBGen::YCSBGeneratorOptions();
  }

  std::atomic<bool> should_stop(false);
  std::thread stat_printer(bg_stat_printer, &work_option, &options,
                           &should_stop);

  Tester tester(work_option);

  auto stats_print_func = [&](std::ostream &log) {
    log << "Timestamp: " << timestamp_ns() << "\n";
    log << "rocksdb.block.cache.data.miss: "
        << options.statistics->getTickerCount(rocksdb::BLOCK_CACHE_DATA_MISS)
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
    log << "rocksdb.bloom.filter.full.true.positive: "
        << options.statistics->getTickerCount(
               rocksdb::BLOOM_FILTER_FULL_TRUE_POSITIVE)
        << "\n";
    log << "rocksdb.memtable.hit: "
        << options.statistics->getTickerCount(rocksdb::MEMTABLE_HIT) << "\n";
    log << "rocksdb.l0.hit: "
        << options.statistics->getTickerCount(rocksdb::GET_HIT_L0) << "\n";
    log << "rocksdb.l1.hit: "
        << options.statistics->getTickerCount(rocksdb::GET_HIT_L1) << "\n";
    log << "rocksdb.rocksdb.l2andup.hit: "
        << options.statistics->getTickerCount(rocksdb::GET_HIT_L2_AND_UP)
        << "\n";
    log << "rocksdb Perf: " << tester.GetRocksdbPerf() << "\n";
    log << "rocksdb IOStats: " << tester.GetRocksdbIOStats() << "\n";

    /* Statistics of router */
    if (router) {
      if (switches & MASK_COUNT_ACCESS_HOT_PER_TIER) {
        auto counters = router->hit_hot_count();
        assert(counters.size() == 2);
        log << "Access hot per tier: " << counters[0] << ' ' << counters[1]
            << "\n";
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
              << ", total " << timer.time().as_secs_double() << " s,\n";
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
          << time.as_secs_double() << " s\n";
    }

    /* Operation counts*/
    log << "operation counts: " << tester.GetOpParseCounts() << "\n";
    log << "notfound counts: " << tester.GetNotFoundCounts() << "\n";
    log << "stat end===" << std::endl;
  };

  auto period_print_stat = [&]() {
    std::ofstream period_stats(db_path / "period_stats");
    while (!should_stop.load()) {
      stats_print_func(period_stats);
      std::this_thread::sleep_for(std::chrono::seconds(1));
    }
  };

  auto print_vc_param_func = [&]() {
    print_vc_param(*router, &work_option, &should_stop);
  };

  std::thread period_print_vc_param_thread(print_vc_param_func);
  std::thread period_print_thread(period_print_stat);

  std::filesystem::path info_json_path = db_path / "info.json";
  std::ofstream info_json_out;
  if (work_option.load) {
    info_json_out = std::ofstream(info_json_path);
    info_json_out << "{" << std::endl;
  } else {
    info_json_out = std::ofstream(info_json_path, std::ios_base::app);
  }
  rusty::sync::Mutex<std::ofstream> info_json(std::move(info_json_out));
  tester.Test(info_json);
  if (work_option.run) {
    auto info_json_locked = info_json.lock();
    *info_json_locked
        << "\t\"IsStablyHot(secs)\": "
        << timers.timer(TimerType::kIsStablyHot).time().as_secs_double()
        << ",\n"
        << "\t\"LowerBound(secs)\": "
        << timers.timer(TimerType::kLowerBound).time().as_secs_double() << ",\n"
        << "\t\"RangeHotSize(secs)\": "
        << timers.timer(TimerType::kRangeHotSize).time().as_secs_double()
        << ",\n"
        << "\t\"NextHot(secs)\": "
        << timers.timer(TimerType::kNextHot).time().as_secs_double() << ",\n";
    rusty::time::Duration hit_level_time;
    size_t num_levels = per_level_timers.len();
    for (size_t level = 0; level < num_levels; ++level) {
      hit_level_time +=
          per_level_timers.timer(level, PerLevelTimerType::kHitLevel).time();
    }
    *info_json_locked << "\t\"HitLevel(secs)\": " << hit_level_time.as_secs_double()
                      << ",\n}" << std::endl;
  }

  should_stop.store(true, std::memory_order_relaxed);
  stats_print_func(std::cerr);

  stat_printer.join();
  period_print_thread.join();
  period_print_vc_param_thread.join();
  delete db;
  delete router;
  delete updater;

  return 0;
}
