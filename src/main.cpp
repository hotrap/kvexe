#include <sys/resource.h>

#include <atomic>
#include <counter_timer_vec.hpp>
#include <sstream>

#include "rocksdb/compaction_router.h"
#include "test.hpp"
#include "viscnts.h"

typedef uint16_t field_size_t;

constexpr size_t MAX_NUM_LEVELS = 8;

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

size_t calc_first_level_in_sd(const rocksdb::Options &options) {
  uint64_t fd_size = options.db_paths[0].target_size;
  size_t level = 0;
  uint64_t level_size = options.max_bytes_for_level_base;
  while (level_size <= fd_size) {
    fd_size -= level_size;
    if (level > 0) {
      level_size *= options.max_bytes_for_level_multiplier;
    }
    level += 1;
  }
  return level;
}

// Solve the equation: x^a + x^{a-1} + ... + x = b
double calc_size_ratio(size_t a, double b) {
  rusty_assert(a > 0);
  double inv_a = 1 / (double)a;
  // Let f(x) = x^a + x^{a-1} + ... + x - b = 0
  // x^a + x^{a-1} + ... + 1 = (x^{a+1} - 1) / (x - 1)
  // x^a + x^{a-1} + ... + x = (x^{a+1} - 1) / (x - 1) - 1
  // (x^{a+1} - 1) / (x - 1) - 1 = b
  // x^{a+1} - 1 = (x - 1) * (b + 1)
  // x^{a+1} - 1 = (b + 1) x - b - 1
  // x^{a+1} - (b + 1) x + b = 0
  // Let g(u) = u^{a+1} - (b + 1) u + maxb
  // g'(u) = (a + 1) u^a - b - 1
  // Let g'(u_min) = 0, then u_min = ((b + 1) / (a + 1)) ^ (1 / a)
  // So x >= ((b + 1) / (a + 1)) ^ (1 / a)
  // x^a + x^{a-1} + ... + x = b > x^a
  // So x < b ^ (1 / a)
  // In conclusion, ((b + 1) / (a + 1)) ^ (1 / a) <= x < b ^ (1 / a)
  double min = pow((b + 1) / (a + 1), inv_a);
  double max = pow(b, inv_a);
  auto f = [a, b](double x) {
    double sum = 0;
    double xa = x;
    for (size_t i = 1; i <= a; ++i) {
      sum += xa;
      xa *= x;
    }
    return sum - b;
  };
  while (max - min >= 0.001) {
    double x = (max + min) / 2;
    if (f(x) > 0) {
      max = x;
    } else {
      min = x;
    }
  }
  return (max + min) / 2;
}
void calc_fd_size_ratio(rocksdb::Options &options, size_t first_level_in_sd,
                        uint64_t max_viscnts_size) {
  options.max_bytes_for_level_multiplier_additional.clear();
  // It seems that L0 and L1 are not affected by
  // options.max_bytes_for_level_multiplier_additional
  if (first_level_in_sd <= 2) return;

  rusty_assert(options.db_paths[0].target_size > max_viscnts_size,
               "max_viscnts_size %" PRIu64 " too large!", max_viscnts_size);
  uint64_t fd_size = options.db_paths[0].target_size - max_viscnts_size;
  double ratio =
      calc_size_ratio(first_level_in_sd - 2,
                      (double)(fd_size - 2 * options.max_bytes_for_level_base) /
                          options.max_bytes_for_level_base);
  // Multiply 0.999 to make room for floating point error
  ratio *= 0.999;
  rusty_assert(options.max_bytes_for_level_multiplier_additional.empty());
  options.max_bytes_for_level_multiplier_additional.push_back(1.0);
  for (size_t i = 2; i < first_level_in_sd; ++i) {
    options.max_bytes_for_level_multiplier_additional.push_back(
        ratio / options.max_bytes_for_level_multiplier);
  }
}
void calc_sd_size_ratio(rocksdb::Options &options, rocksdb::DB *db,
                        size_t last_level_in_fd, uint64_t last_level_in_fd_size,
                        uint64_t hot_set_size_limit) {
  std::string str;
  db->GetProperty(rocksdb::DB::Properties::kLevelStats, &str);
  std::istringstream in(str);
  // The first two lines are headers.
  size_t lines_to_skip = 2 + last_level_in_fd + 1;
  while (in && lines_to_skip) {
    --lines_to_skip;
    while (in && in.get() != '\n')
      ;
  }
  if (!in) return;

  size_t last_level = last_level_in_fd;
  uint64_t sd_level_size = 0;
  while (in) {
    size_t level;
    size_t num_files;
    size_t size;
    in >> level >> num_files >> size;
    if (size == 0) break;
    last_level = level;
    sd_level_size += size;
  }
  sd_level_size *= 1048576;
  if (last_level <= last_level_in_fd + 1) return;
  // unlikely
  if (sd_level_size <= last_level_in_fd_size) return;

  rusty_assert(last_level_in_fd > 1, "Not implemented yet!");
  uint64_t last_level_in_fd_effective_size =
      last_level_in_fd_size - hot_set_size_limit;
  size_t a = last_level - last_level_in_fd;
  double b = (double)sd_level_size / last_level_in_fd_effective_size;
  double sd_ratio = calc_size_ratio(a, b);
  size_t level = last_level_in_fd + 1;
  if (sd_ratio > options.max_bytes_for_level_multiplier) {
    do {
      last_level += 1;
      a += 1;
      sd_ratio = calc_size_ratio(a, b);
    } while (sd_ratio > options.max_bytes_for_level_multiplier);
  }
  // The first level in SD shouldn't be smaller than the last level in FD.
  if (last_level_in_fd_effective_size * sd_ratio <
      last_level_in_fd_size * 1.01) {
    double ratio_additional = 1.01 / options.max_bytes_for_level_multiplier;
    options.max_bytes_for_level_multiplier_additional.push_back(
        ratio_additional);
    uint64_t first_level_in_sd_size = last_level_in_fd_size *
                                      options.max_bytes_for_level_multiplier *
                                      ratio_additional;
    a -= 1;
    b = (double)(sd_level_size - first_level_in_sd_size) /
        first_level_in_sd_size;
    sd_ratio = calc_size_ratio(a, b);
    level += 1;

    if (sd_ratio > options.max_bytes_for_level_multiplier) {
      do {
        last_level += 1;
        a += 1;
        sd_ratio = calc_size_ratio(a, b);
      } while (sd_ratio > options.max_bytes_for_level_multiplier);
    }
  } else {
    options.max_bytes_for_level_multiplier_additional.push_back(
        last_level_in_fd_effective_size * sd_ratio / last_level_in_fd_size /
        options.max_bytes_for_level_multiplier);
    level += 1;
  }
  sd_ratio /= options.max_bytes_for_level_multiplier;
  for (; level < last_level; ++level) {
    options.max_bytes_for_level_multiplier_additional.push_back(sd_ratio);
  }
  options.max_bytes_for_level_multiplier_additional.push_back(100.0);
}
bool should_update_max_bytes_for_level_multiplier_additional(
    const std::vector<double> &ori, const std::vector<double> &cur) {
  if (ori.size() != cur.size()) {
    rusty_assert(ori.size() < cur.size());
    return true;
  }
  for (size_t i = 0; i < ori.size(); ++i) {
    if (cur[i] < ori[i] * 0.99) return true;
    if (cur[i] > ori[i] * 1.01) return true;
  }
  return false;
}

double MaxBytesMultiplerAdditional(const rocksdb::Options &options, int level) {
  if (level >= static_cast<int>(
                   options.max_bytes_for_level_multiplier_additional.size())) {
    return 1;
  }
  return options.max_bytes_for_level_multiplier_additional[level];
}

std::vector<std::pair<uint64_t, uint32_t>> predict_level_assignment(
    const rocksdb::Options &options) {
  std::vector<std::pair<uint64_t, uint32_t>> ret;
  uint32_t p = 0;
  size_t level = 0;
  assert(!options.db_paths.empty());

  // size remaining in the most recent path
  uint64_t current_path_size = options.db_paths[0].target_size;

  uint64_t level_size;
  size_t cur_level = 0;

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
      rusty_assert_eq(ret.size(), level);
      ret.emplace_back(level_size, p);
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
  rusty_assert_eq(ret.size(), level);
  ret.emplace_back(level_size, p);
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
  kEnd,
};
constexpr size_t PER_LEVEL_TIMER_NUM =
    static_cast<size_t>(PerLevelTimerType::kEnd);
const char *per_level_timer_names[] = {};
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
                size_t max_hot_set_size, size_t min_hot_set_size,
                size_t bloom_bfk, 
                bool enable_sampling)
      : switches_(switches),
        vc_(VisCnts::New(ucmp, dir.c_str(), init_hot_set_size, max_hot_set_size,
                         min_hot_set_size, max_viscnts_size, bloom_bfk)),
        tier0_last_level_(tier0_last_level),
        count_access_hot_per_tier_{0, 0},
        count_access_fd_hot_(0),
        count_access_fd_cold_(0),
        enable_sampling_(enable_sampling) {
    for (size_t i = 0; i < MAX_NUM_LEVELS; ++i) {
      level_hits_[i].store(0, std::memory_order_relaxed);
    }
  }
  const char *Name() const override { return "RouterVisCnts"; }
  size_t Tier(int level) override {
    if (level <= tier0_last_level_) {
      return 0;
    } else {
      return 1;
    }
  }
  void HitLevel(int level, rocksdb::Slice key) override {
    if (get_key_hit_level_out().has_value()) {
      get_key_hit_level_out().value()
          << timestamp_ns() << ' ' << key.ToString() << ' ' << level << '\n';
    }
    if (level < 0) level = 0;
    rusty_assert((size_t)level < MAX_NUM_LEVELS);
    level_hits_[level].fetch_add(1, std::memory_order_relaxed);

    if (switches_ & MASK_COUNT_ACCESS_HOT_PER_TIER) {
      size_t tier = Tier(level);
      bool is_hot = vc_.IsHot(key);
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
  void Access(rocksdb::Slice key, size_t vlen) override {
    thread_local static std::optional<std::mt19937> rgen;
    auto guard = timers.timer(TimerType::kAccess).start();
    double rate =
        count_access_hot_per_tier_[0].load(std::memory_order_relaxed) /
        (double)(count_access_hot_per_tier_[0].load(std::memory_order_relaxed) +
                 count_access_hot_per_tier_[1].load(std::memory_order_relaxed));
    if (rate > 0.95 && enable_sampling_) {
      double A = (0.95 / rate);
      if (!rgen) {
        rgen = std::mt19937(std::random_device()());
      }
      std::uniform_real_distribution<> dis(0, 1);
      if (dis(rgen.value()) < A) {
        vc_.Access(key, vlen);
      }
    } else {
      vc_.Access(key, vlen);
    }
  }

  bool IsHot(rocksdb::Slice key) override {
    auto guard = timers.timer(TimerType::kIsHot).start();
    return vc_.IsHot(key);
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

  VisCnts &get_vc() { return vc_; }

 private:
  const uint64_t switches_;
  VisCnts vc_;
  int tier0_last_level_;

  std::atomic<uint64_t> level_hits_[MAX_NUM_LEVELS];
  std::atomic<uint64_t> count_access_hot_per_tier_[2];
  std::atomic<uint64_t> count_access_fd_hot_;
  std::atomic<uint64_t> count_access_fd_cold_;
  bool enable_sampling_{false};
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

  double AddPeriodData(const std::vector<size_t> &hr) {
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

class AutoTuner {
 public:
  AutoTuner(const WorkOptions &work_options, rocksdb::Options &options,
            size_t first_level_in_sd, uint64_t max_vc_hot_set_size,
            uint64_t min_vc_hot_set_size, size_t wait_time_ns,
            RouterVisCnts &router)
      : work_options_(work_options),
        options_(options),
        first_level_in_sd_(first_level_in_sd),
        wait_time_ns_(wait_time_ns),
        max_vc_hot_set_size_(max_vc_hot_set_size),
        min_vc_hot_set_size_(min_vc_hot_set_size),
        router_(router),
        log_(work_options_.db_path / "vc_log") {
    th_ = std::thread([&]() { update_thread(); });
  }

  ~AutoTuner() { Stop(); }

  void Stop() {
    stop_signal_ = true;
    th_.join();
  }

 private:
  void update_thread() {
    VisCnts &vc = router_.get_vc();
    const uint64_t initial_max_hot_set_size = vc.GetMaxHotSetSizeLimit();
    std::cerr << "Initial max hot set size: " << initial_max_hot_set_size
              << std::endl;

    const uint64_t initial_hot_set_size_limit = vc.GetHotSetSizeLimit();
    std::cerr << "Initial hot set size limit: " << initial_hot_set_size_limit
              << std::endl;

    uint64_t phy_size_limit = vc.GetPhySizeLimit();
    std::cerr << "Initial physical size limit: " << phy_size_limit << std::endl;

    const size_t last_level_in_fd = first_level_in_sd_ - 1;
    std::vector<double> ori_multiplier_additional =
        options_.max_bytes_for_level_multiplier_additional;
    bool warming_up = true;
    bool first = true;
    while (!stop_signal_) {
      std::this_thread::sleep_for(std::chrono::nanoseconds(wait_time_ns_));
      if (stop_signal_) {
        break;
      }
      if (router_.get_vc().DecayCount() > 10) {
        if (first) {
          first = false;
          warming_up = false;
          vc.SetMinHotSetSizeLimit(min_vc_hot_set_size_);
        }
        double hs_step = max_vc_hot_set_size_ / 20.0;
        uint64_t real_phy_size = vc.GetRealPhySize();
        uint64_t real_hot_set_size = vc.GetRealHotSetSize();
        std::cerr << "real_phy_size " << real_phy_size << '\n'
                  << "real_hot_set_size " << real_hot_set_size << '\n';
        auto rate = real_phy_size / (double)real_hot_set_size;
        auto delta =
            rate * hs_step;  // std::max<size_t>(rate * hs_step, (64 << 20));
        phy_size_limit = real_phy_size + delta;
        std::cerr << "rate " << rate << std::endl;
        router_.get_vc().SetPhysicalSizeLimit(phy_size_limit);
        std::cerr << "Update physical size limit: " << phy_size_limit
                  << std::endl;
      }
      calc_fd_size_ratio(options_, first_level_in_sd_, phy_size_limit);
      rusty_assert(first_level_in_sd_ > 0);
      uint64_t last_level_in_fd_size =
          predict_level_assignment(options_)[last_level_in_fd].first;
      uint64_t min_effective_size_of_last_level_in_fd =
          last_level_in_fd_size / options_.max_bytes_for_level_multiplier;
      // to avoid making the size of the first level in the slow disk too small
      uint64_t max_hot_set_size =
          last_level_in_fd_size - min_effective_size_of_last_level_in_fd;
      if (warming_up) {
        max_hot_set_size = std::min(max_hot_set_size, initial_max_hot_set_size);
      } else {
        max_hot_set_size = std::min(max_hot_set_size, max_vc_hot_set_size_);
      }
      if (vc.GetMaxHotSetSizeLimit() != max_hot_set_size) {
        std::cerr << "Update max hot set size limit: " << max_hot_set_size
                  << std::endl;
        vc.SetMaxHotSetSizeLimit(max_hot_set_size);
      }

      uint64_t hot_set_size_limit = router_.get_vc().GetHotSetSizeLimit();
      if (warming_up) {
        rusty_assert_eq(hot_set_size_limit, initial_hot_set_size_limit);
      } else {
        std::cerr << "hot set size limit: " << hot_set_size_limit << std::endl;
      }
      calc_sd_size_ratio(options_, work_options_.db, last_level_in_fd,
                         last_level_in_fd_size, hot_set_size_limit);
      if (should_update_max_bytes_for_level_multiplier_additional(
              ori_multiplier_additional,
              options_.max_bytes_for_level_multiplier_additional)) {
        ori_multiplier_additional.assign(
            options_.max_bytes_for_level_multiplier_additional.begin(),
            options_.max_bytes_for_level_multiplier_additional.end());
        std::ostringstream out;
        for (size_t i = 0; i < ori_multiplier_additional.size(); ++i) {
          out << ori_multiplier_additional[i];
          if (i != ori_multiplier_additional.size()) {
            out << ':';
          }
        }
        std::string str = out.str();
        std::cerr << "Update max_bytes_for_level_multiplier_additional: " << str
                  << std::endl;
        work_options_.db->SetOptions(
            {{"max_bytes_for_level_multiplier_additional", str}});
      }
    }
  }

  const WorkOptions &work_options_;
  rocksdb::Options &options_;
  const size_t first_level_in_sd_;

  ssize_t wait_time_ns_;
  uint64_t max_vc_hot_set_size_;
  uint64_t min_vc_hot_set_size_;
  RouterVisCnts &router_;
  std::ofstream log_;

  bool stop_signal_{false};
  std::thread th_;
};

void print_vc_param(RouterVisCnts &router, WorkOptions *work_options,
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
                "get-cpu-nanos delete-cpu-nanos";
  uint64_t value;
  bool has_viscnts_compaction_thread_cpu_nanos =
      router->get_viscnts_int_property("viscnts.compaction.thread.cpu.nanos",
                                       &value);
  if (has_viscnts_compaction_thread_cpu_nanos) {
    timers_out << " viscnts.compaction.thread.cpu.nanos";
  }
  bool has_viscnts_flush_thread_cpu_nanos = router->get_viscnts_int_property(
      "viscnts.flush.thread.cpu.nanos", &value);
  if (has_viscnts_flush_thread_cpu_nanos) {
    timers_out << " viscnts.flush.thread.cpu.nanos";
  }
  bool has_viscnts_decay_thread_cpu_nanos = router->get_viscnts_int_property(
      "viscnts.decay.thread.cpu.nanos", &value);
  if (has_viscnts_decay_thread_cpu_nanos) {
    timers_out << " viscnts.decay.thread.cpu.nanos";
  }
  bool has_viscnts_compaction_cpu_nanos =
      router->get_viscnts_int_property("viscnts.compaction.cpu.nanos", &value);
  if (has_viscnts_compaction_cpu_nanos) {
    timers_out << " viscnts.compaction.cpu.nanos";
  }
  bool has_viscnts_flush_cpu_nanos =
      router->get_viscnts_int_property("viscnts.flush.cpu.nanos", &value);
  if (has_viscnts_flush_cpu_nanos) {
    timers_out << " viscnts.flush.cpu.nanos";
  }
  bool has_viscnts_decay_scan_cpu_nanos =
      router->get_viscnts_int_property("viscnts.decay.scan.cpu.nanos", &value);
  if (has_viscnts_decay_scan_cpu_nanos) {
    timers_out << " viscnts.decay.scan.cpu.nanos";
  }
  bool has_viscnts_decay_write_cpu_nanos =
      router->get_viscnts_int_property("viscnts.decay.write.cpu.nanos", &value);
  if (has_viscnts_decay_write_cpu_nanos) {
    timers_out << " viscnts.decay.write.cpu.nanos";
  }
  timers_out << std::endl;

  std::ofstream rand_read_bytes_out(db_path / "rand-read-bytes");

  // Stats of hotrap

  std::ofstream promoted_or_retained_out(db_path /
                                         "promoted-or-retained-bytes");
  promoted_or_retained_out
      << "Timestamp(ns) by-flush 2fdlast 2sdfront retained\n";

  std::ofstream not_promoted_bytes_out(db_path / "not-promoted-bytes");
  not_promoted_bytes_out << "Timestamp(ns) not-hot has-newer-version\n";

  std::ofstream num_accesses_out(db_path / "num-accesses");

  std::ofstream viscnts_io_out(db_path / "viscnts-io");
  viscnts_io_out << "Timestamp(ns) read write\n";

  std::ofstream viscnts_sizes(db_path / "viscnts-sizes");
  viscnts_sizes << "Timestamp(ns) real-phy-size real-hot-size\n";

  auto stats = options->statistics;

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
    timers_out << timestamp << ' ' << compaction_cpu_micros << ' '
               << put_cpu_nanos.load(std::memory_order_relaxed) << ' '
               << get_cpu_nanos.load(std::memory_order_relaxed) << ' '
               << delete_cpu_nanos.load(std::memory_order_relaxed);
    if (has_viscnts_compaction_thread_cpu_nanos) {
      rusty_assert(router->get_viscnts_int_property(
          "viscnts.compaction.thread.cpu.nanos", &value));
      timers_out << ' ' << value;
    }
    if (has_viscnts_flush_thread_cpu_nanos) {
      rusty_assert(router->get_viscnts_int_property(
          "viscnts.flush.thread.cpu.nanos", &value));
      timers_out << ' ' << value;
    }
    if (has_viscnts_decay_thread_cpu_nanos) {
      rusty_assert(router->get_viscnts_int_property(
          "viscnts.decay.thread.cpu.nanos", &value));
      timers_out << ' ' << value;
    }
    if (has_viscnts_compaction_cpu_nanos) {
      rusty_assert(router->get_viscnts_int_property(
          "viscnts.compaction.cpu.nanos", &value));
      timers_out << ' ' << value;
    }
    if (has_viscnts_flush_cpu_nanos) {
      rusty_assert(
          router->get_viscnts_int_property("viscnts.flush.cpu.nanos", &value));
      timers_out << ' ' << value;
    }
    if (has_viscnts_decay_scan_cpu_nanos) {
      rusty_assert(router->get_viscnts_int_property(
          "viscnts.decay.scan.cpu.nanos", &value));
      timers_out << ' ' << value;
    }
    if (has_viscnts_decay_write_cpu_nanos) {
      rusty_assert(router->get_viscnts_int_property(
          "viscnts.decay.write.cpu.nanos", &value));
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
        << stats->getTickerCount(rocksdb::RETAINED_BYTES) << std::endl;

    not_promoted_bytes_out
        << timestamp << ' '
        << stats->getTickerCount(rocksdb::ACCESSED_COLD_BYTES) << ' '
        << stats->getTickerCount(rocksdb::HAS_NEWER_VERSION_BYTES) << std::endl;

    num_accesses_out << timestamp;
    auto level_hits = router->level_hits();
    for (size_t hits : level_hits) {
      num_accesses_out << ' ' << hits;
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

    VisCnts &vc = router->get_vc();
    viscnts_sizes << timestamp << ' ' << vc.GetRealPhySize() << ' '
                  << vc.GetRealHotSetSize() << std::endl;

    auto sleep_time =
        next_begin.checked_duration_since(rusty::time::Instant::now());
    if (sleep_time.has_value()) {
      std::this_thread::sleep_for(
          std::chrono::nanoseconds(sleep_time.value().as_nanos()));
    }
    next_begin += interval;
  }
}

void print_other_stats(std::ostream &log, const rocksdb::Options &options,
                       Tester &tester) {
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
      << options.statistics->getTickerCount(rocksdb::BLOOM_FILTER_FULL_POSITIVE)
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
      << options.statistics->getTickerCount(rocksdb::GET_HIT_L2_AND_UP) << "\n";
  log << "leader write count: "
      << options.statistics->getTickerCount(rocksdb::LEADER_WRITE_COUNT)
      << '\n';
  log << "non leader write count: "
      << options.statistics->getTickerCount(rocksdb::NON_LEADER_WRITE_COUNT)
      << '\n';

  log << "Promotion cache hits: "
      << options.statistics->getTickerCount(rocksdb::GET_HIT_PROMOTION_CACHE)
      << "\n";
  log << "rocksdb Perf: " << tester.GetRocksdbPerf() << "\n";
  log << "rocksdb IOStats: " << tester.GetRocksdbIOStats() << "\n";

  print_timers(log);

  /* Operation counts*/
  log << "notfound counts: " << tester.GetNotFoundCounts() << "\n";
  log << "stat end===" << std::endl;
}

int main(int argc, char **argv) {
  std::ios::sync_with_stdio(false);
  std::cin.tie(0);
  std::cout.tie(0);

  rocksdb::BlockBasedTableOptions table_options;
  rocksdb::Options options;
  WorkOptions work_options;

  namespace po = boost::program_options;
  po::options_description desc("Available options");
  std::string format;
  std::string arg_switches;
  size_t num_threads;

  std::string arg_db_path;
  std::string arg_db_paths;
  std::string viscnts_path_str;
  size_t cache_size;
  int64_t load_phase_rate_limit;

  double arg_max_hot_set_size;
  double arg_max_viscnts_size;
  int compaction_pri;
  int ralt_bloom_bpk;

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
  desc.add_options()("num_threads", po::value(&num_threads)->default_value(1),
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
  desc.add_options()("block_size", po::value<size_t>(&table_options.block_size),
                     "Default: 4096");
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
  desc.add_options()("max_viscnts_size",
                     po::value<double>(&arg_max_viscnts_size)->required(),
                     "Max physical size of viscnts in bytes");
  desc.add_options()("viscnts_path",
                     po::value<std::string>(&viscnts_path_str)->required(),
                     "Path to VisCnts");
  desc.add_options()("compaction_pri,p",
                     po::value<int>(&compaction_pri)->required(),
                     "Method to pick SST to compact (rocksdb::CompactionPri)");

  desc.add_options()("enable_auto_tuning", "enable auto-tuning");

  desc.add_options()("enable_sampling", "enable_sampling");

  desc.add_options()("ralt_bloom_bpk", po::value<int>(&ralt_bloom_bpk)->default_value(10), "The number of bits per key in RALT bloom filter.");

  po::variables_map vm;
  po::store(po::parse_command_line(argc, argv, desc), vm);
  if (vm.count("help")) {
    std::cerr << desc << std::endl;
    return 1;
  }
  po::notify(vm);

  uint64_t hot_set_size_limit = arg_max_hot_set_size;
  uint64_t max_viscnts_size = arg_max_viscnts_size;

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
  std::filesystem::path viscnts_path(viscnts_path_str);
  options.db_paths = decode_db_paths(arg_db_paths);
  options.compaction_pri = static_cast<rocksdb::CompactionPri>(compaction_pri);
  options.statistics = rocksdb::CreateDBStatistics();
  options.compression = rocksdb::CompressionType::kNoCompression;
  // Doesn't make sense for tiered storage
  options.level_compaction_dynamic_level_bytes = false;

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

  size_t first_level_in_sd = calc_first_level_in_sd(options);
  calc_fd_size_ratio(options, first_level_in_sd, max_viscnts_size);

  auto ret = predict_level_assignment(options);
  rusty_assert_eq(ret.size() - 1, first_level_in_sd);

  RouterVisCnts *router = nullptr;
  if (first_level_in_sd != 0) {
    router = new RouterVisCnts(options.comparator, viscnts_path_str,
                               first_level_in_sd - 1, hot_set_size_limit,
                               max_viscnts_size, switches, hot_set_size_limit,
                               hot_set_size_limit, ralt_bloom_bpk, vm.count("enable_sampling"));

    options.compaction_router = router;
  }

  rocksdb::DB *db;
  if (work_options.load) {
    std::cerr << "Emptying directories\n";
    empty_directory(db_path);
    for (auto path : options.db_paths) {
      empty_directory(path.path);
    }
    for (size_t level = 0; level < first_level_in_sd; ++level) {
      auto p = ret[level].second;
      std::cerr << level << ' ' << options.db_paths[p].path << ' '
                << ret[level].first << std::endl;
    }
    auto p = ret[first_level_in_sd].second;
    std::cerr << first_level_in_sd << "+ " << options.db_paths[p].path << ' '
              << ret[first_level_in_sd].first << std::endl;
    if (options.db_paths.size() == 1) {
      first_level_in_sd = 100;
    }
    std::ofstream(db_path / "first-level-in-sd")
        << first_level_in_sd << std::endl;

    std::cerr << "Creating database\n";
    options.create_if_missing = true;
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

  std::atomic<uint64_t> progress(0);
  std::atomic<uint64_t> progress_get(0);

  work_options.db = db;
  work_options.switches = switches;
  work_options.db_path = db_path;
  work_options.progress = &progress;
  work_options.progress_get = &progress_get;
  work_options.num_threads = num_threads;
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

  std::atomic<bool> should_stop(false);
  std::thread stat_printer(bg_stat_printer, &work_options, &options,
                           &should_stop);

  Tester tester(work_options);

  AutoTuner *autotuner = nullptr;
  if (vm.count("enable_auto_tuning") && router) {
    autotuner =
        new AutoTuner(work_options, options, first_level_in_sd,
                      options.db_paths[0].target_size * 0.7,
                      options.db_paths[0].target_size * 0.05, 20e9, *router);
  }

  auto period_print_stat = [&]() {
    std::ofstream period_stats(db_path / "period_stats");
    while (!should_stop.load()) {
      print_other_stats(period_stats, options, tester);
      std::this_thread::sleep_for(std::chrono::seconds(1));
    }
  };

  auto print_vc_param_func = [&]() {
    print_vc_param(*router, &work_options, &should_stop);
  };

  std::thread period_print_vc_param_thread(print_vc_param_func);
  std::thread period_print_thread(period_print_stat);

  std::filesystem::path info_json_path = db_path / "info.json";
  std::ofstream info_json_out;
  if (work_options.load) {
    info_json_out = std::ofstream(info_json_path);
    info_json_out << "{" << std::endl;
  } else {
    info_json_out = std::ofstream(info_json_path, std::ios_base::app);
  }
  rusty::sync::Mutex<std::ofstream> info_json(std::move(info_json_out));
  tester.Test(info_json);
  if (work_options.run) {
    auto info_json_locked = info_json.lock();
    *info_json_locked
        << "\t\"IsHot(secs)\": "
        << timers.timer(TimerType::kIsHot).time().as_secs_double() << ",\n"
        << "\t\"LowerBound(secs)\": "
        << timers.timer(TimerType::kLowerBound).time().as_secs_double() << ",\n"
        << "\t\"RangeHotSize(secs)\": "
        << timers.timer(TimerType::kRangeHotSize).time().as_secs_double()
        << ",\n"
        << "\t\"NextHot(secs)\": "
        << timers.timer(TimerType::kNextHot).time().as_secs_double() << "\n}"
        << std::endl;
  }

  should_stop.store(true, std::memory_order_relaxed);

  print_other_stats(std::cerr, options, tester);

  /* Statistics of router */
  if (router) {
    if (tester.work_options().switches & MASK_COUNT_ACCESS_HOT_PER_TIER) {
      auto counters = router->hit_hot_count();
      assert(counters.size() == 2);
      std::cerr << "Access hot per tier: " << counters[0] << ' ' << counters[1]
                << "\nAccess FD hot: " << router->count_access_fd_hot()
                << "\nAccess FD cold: " << router->count_access_fd_cold()
                << '\n';
    }
  }

  stat_printer.join();
  period_print_thread.join();
  period_print_vc_param_thread.join();
  delete autotuner;
  delete db;
  delete router;

  return 0;
}
