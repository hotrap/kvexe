#include <counter_timer_vec.hpp>

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
  RouterVisCnts(const rocksdb::Comparator *ucmp, std::filesystem::path dir,
                int tier0_last_level, size_t max_hot_set_size,
                uint64_t switches)
      : switches_(switches),
        vc_(VisCnts::New(ucmp, dir.c_str(), max_hot_set_size)),
        tier0_last_level_(tier0_last_level),
        new_iter_cnt_(0),
        count_access_hot_per_tier_{0, 0} {}
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
      if (vc_.IsHot(key)) count_access_hot_per_tier_[tier].fetch_add(1);
    }

    auto guard =
        per_level_timers.timer(level, PerLevelTimerType::kAccess).start();
    vc_.Access(key, vlen);
    if (get_key_hit_level_out().has_value()) {
      get_key_hit_level_out().value() << key.ToString() << ' ' << level << '\n';
    }
  }

  bool IsStablyHot(rocksdb::Slice key) override { return vc_.IsStablyHot(key); }
  // The returned pointer will stay valid until the next call to Seek or
  // NextHot with this iterator
  rocksdb::CompactionRouter::Iter LowerBound(rocksdb::Slice key) override {
    new_iter_cnt_.fetch_add(1, std::memory_order_relaxed);
    return vc_.LowerBound(key);
  }
  size_t RangeHotSize(rocksdb::Slice smallest,
                      rocksdb::Slice largest) override {
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
};

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
  auto mem_path = db_path / "mem";
  std::ofstream(mem_path) << "Timestamp(ns) RSS(KB)\n";

  std::ofstream promoted_2sdlast_out(db_path / "promoted-2sdlast-bytes");
  promoted_2sdlast_out << "Timestamp(ns) num-bytes\n";
  std::ofstream promoted_flush_out(db_path / "promoted-flush-bytes");
  promoted_flush_out << "Timestamp(ns) num-bytes\n";

  std::ofstream num_accesses_out(db_path / "num-accesses");

  while (!should_stop->load(std::memory_order_relaxed)) {
    auto timestamp = timestamp_ns();

    auto value = progress->load(std::memory_order_relaxed);
    progress_out << timestamp << ' ' << value << std::endl;

    std::ofstream(mem_path, std::ios_base::app) << timestamp << ' ';
    std::system(("ps -q " + std::to_string(getpid()) +
                 " -o rss | tail -n 1 >> " + mem_path.c_str())
                    .c_str());

    auto promoted_2sdlast_bytes =
        options->statistics->getTickerCount(rocksdb::PROMOTED_2SDLAST_BYTES);
    promoted_2sdlast_out << timestamp << ' ' << promoted_2sdlast_bytes
                         << std::endl;

    auto promoted_flush_bytes =
        options->statistics->getTickerCount(rocksdb::PROMOTED_FLUSH_BYTES);
    promoted_flush_out << timestamp << ' ' << promoted_flush_bytes << std::endl;

    size_t num_level = per_level_timers.len();
    num_accesses_out << timestamp;
    for (size_t level = 0; level < num_level; ++level) {
      num_accesses_out
          << ' '
          << per_level_timers.timer(level, PerLevelTimerType::kAccess).count();
    }
    num_accesses_out << std::endl;

    std::this_thread::sleep_for(std::chrono::seconds(1));
  }
}

int main(int argc, char **argv) {
  std::ios::sync_with_stdio(false);
  std::cin.tie(0);
  std::cout.tie(0);

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
  desc.add_options()("help", "Print help message");
  desc.add_options()("cleanup,c", "Empty the directories first.");
  desc.add_options()("max_background_jobs", po::value<int>(), "");
  desc.add_options()("level0_file_num_compaction_trigger", po::value<int>(),
                     "Number of files in level-0 when compactions start");
  desc.add_options()("format,f",
                     po::value<std::string>(&format)->default_value("ycsb"),
                     "Trace format: plain/ycsb");
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
  desc.add_options()("enable_fast_process",
                     "Enable fast process including ignoring kNotFound and "
                     "pushing operations in one channel.");
  desc.add_options()("enable_fast_generator", "Enable fast generator");
  desc.add_options()("workload_file", po::value<std::string>(),
                     "Workload file used in built-in generator");
  desc.add_options()("export_key_only_trace",
                     "Export key-only trace generated by built-in generator.");

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

  rocksdb::BlockBasedTableOptions table_options;
  table_options.block_cache = rocksdb::NewLRUCache(cache_size);
  table_options.filter_policy.reset(rocksdb::NewBloomFilterPolicy(10, false));
  options.table_factory.reset(
      rocksdb::NewBlockBasedTableFactory(table_options));

  if (vm.count("max_background_jobs")) {
    options.max_background_jobs = vm["max_background_jobs"].as<int>();
  }

  if (vm.count("level0_file_num_compaction_trigger")) {
    options.level0_file_num_compaction_trigger =
        vm["level0_file_num_compaction_trigger"].as<int>();
  }

  if (vm.count("cleanup")) {
    std::cerr << "Emptying directories\n";
    empty_directory(db_path);
    for (auto path : options.db_paths) {
      empty_directory(path.path);
    }
    empty_directory(viscnts_path_str);
  }

  auto ret = predict_level_assignment(options);
  rusty_assert(ret.size() > 0);
  size_t first_level_in_cd = ret.size() - 1;
  if (max_hot_set_size != 0) {
    for (;;) {
      // It seems that L0 and L1 are not affected by
      // options.max_bytes_for_level_multiplier_additional
      rusty_assert(first_level_in_cd >= 3);
      size_t last_level_in_sd = first_level_in_cd - 1;
      options.max_bytes_for_level_multiplier_additional.clear();
      for (size_t level = 1; level < last_level_in_sd; ++level) {
        options.max_bytes_for_level_multiplier_additional.push_back(1.0);
      }
      options.max_bytes_for_level_multiplier_additional.push_back(
          1 + (double)max_hot_set_size / ret[last_level_in_sd].first);
      ret = predict_level_assignment(options);
      if (ret.size() - 1 == first_level_in_cd) break;
      rusty_assert(ret.size() - 1 < first_level_in_cd);
      first_level_in_cd = ret.size() - 1;
    }
    std::cerr << "options.max_bytes_for_level_multiplier_additional: [";
    for (double x : options.max_bytes_for_level_multiplier_additional) {
      std::cerr << x << ',';
    }
    std::cerr << "]\n";
  }
  for (size_t level = 0; level < first_level_in_cd; ++level) {
    std::cerr << level << ' ' << ret[level].second << ' ' << ret[level].first
              << std::endl;
  }
  std::cerr << first_level_in_cd << "+ " << ret[first_level_in_cd].second << ' '
            << ret[first_level_in_cd].first << std::endl;
  std::ofstream(db_path / "first-level-in-cd")
      << first_level_in_cd << std::endl;

  RouterVisCnts *router = nullptr;
  if (first_level_in_cd != 0) {
    router =
        new RouterVisCnts(options.comparator, viscnts_path_str,
                          first_level_in_cd - 1, max_hot_set_size, switches);
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

  WorkOptions work_option;
  work_option.db = db;
  work_option.switches = switches;
  work_option.db_path = db_path;
  work_option.progress = &progress;
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
      log << "New iterator count: " << router->new_iter_cnt() << "\n";
      if (switches & MASK_COUNT_ACCESS_HOT_PER_TIER) {
        auto counters = router->hit_count();
        assert(counters.size() == 2);
        log << "Access hot per tier: " << counters[0] << ' ' << counters[1]
            << "\n";
      }

      size_t num_tiers = per_tier_timers.len();
      for (size_t tier = 0; tier < num_tiers; ++tier) {
        log << "Tier timers: {tier: " << tier << ", timers: [\n";
        const auto &timers = per_tier_timers.timers(tier);
        size_t num_types = timers.len();
        for (size_t type = 0; type < num_types; ++type) {
          const auto &timer = timers.timer(type);
          log << per_tier_timer_names[type] << ": count " << timer.count()
              << ", total " << timer.time().as_secs_double() << " s,\n";
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
  rusty_assert(db->GetProperty("rocksdb.stats", &rocksdb_stats));
  std::ofstream(db_path / "rocksdb-stats.txt") << rocksdb_stats;

  stat_printer.join();
  period_print_thread.join();
  delete db;
  delete router;

  return 0;
}
