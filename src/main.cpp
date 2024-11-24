#include <autotuner.h>
#include <sys/resource.h>

#include <atomic>
#include <counter_timer_vec.hpp>

#include "ralt.h"
#include "rocksdb/ralt.h"
#include "test.hpp"

static inline void empty_directory(std::filesystem::path dir_path) {
  for (auto &path : std::filesystem::directory_iterator(dir_path)) {
    std::filesystem::remove_all(path);
  }
}

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

class RaltWrapper : public RALT {
 public:
  RaltWrapper(const rocksdb::Comparator *ucmp, std::filesystem::path dir,
              int tier0_last_level, size_t init_hot_set_size,
              size_t max_ralt_size, uint64_t switches, size_t max_hot_set_size,
              size_t min_hot_set_size, size_t bloom_bfk)
      : RALT(ucmp, dir.c_str(), init_hot_set_size, max_hot_set_size,
             min_hot_set_size, max_ralt_size, bloom_bfk),
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
  auto &ralt = *static_cast<RaltWrapper *>(options->ralt);

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
      << "Timestamp(ns) by-flush 2fdlast 2sdfront retained\n";

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
        << stats->getTickerCount(rocksdb::RETAINED_BYTES) << std::endl;

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
    rusty_assert(ralt.GetIntProperty(RALT::Properties::kReadBytes, &ralt_read));
    uint64_t ralt_write;
    rusty_assert(
        ralt.GetIntProperty(RALT::Properties::kWriteBytes, &ralt_write));
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
  desc.add_options()("ralt_bloom_bpk",
                     po::value<int>(&ralt_bloom_bpk)->default_value(14),
                     "The number of bits per key in RALT bloom filter.");

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

  size_t first_level_in_sd = calc_first_level_in_sd(options);
  calc_fd_size_ratio(options, first_level_in_sd, max_ralt_size);

  auto level_size_path_id = predict_level_assignment(options);
  rusty_assert_eq(level_size_path_id.size() - 1, first_level_in_sd);

  for (size_t level = 0; level < first_level_in_sd; ++level) {
    auto p = level_size_path_id[level].second;
    std::cerr << level << ' ' << options.db_paths[p].path << ' '
              << level_size_path_id[level].first << std::endl;
  }
  auto p = level_size_path_id[first_level_in_sd].second;
  std::cerr << first_level_in_sd << "+ " << options.db_paths[p].path << ' '
            << level_size_path_id[first_level_in_sd].first << std::endl;
  if (options.db_paths.size() == 1) {
    first_level_in_sd = 100;
  }
  auto first_level_in_sd_path = db_path / "first-level-in-sd";
  if (std::filesystem::exists(first_level_in_sd_path)) {
    std::ifstream first_level_in_sd_in(first_level_in_sd_path);
    rusty_assert(first_level_in_sd_in);
    std::string first_level_in_sd_stored;
    std::getline(first_level_in_sd_in, first_level_in_sd_stored);
    rusty_assert_eq((size_t)std::atoi(first_level_in_sd_stored.c_str()),
                    first_level_in_sd);
  } else {
    std::ofstream(first_level_in_sd_path) << first_level_in_sd << std::endl;
  }

  RaltWrapper *ralt = nullptr;
  if (first_level_in_sd != 0) {
    ralt = new RaltWrapper(options.comparator, ralt_path_str,
                           first_level_in_sd - 1, hot_set_size_limit,
                           max_ralt_size, switches, hot_set_size_limit,
                           hot_set_size_limit, ralt_bloom_bpk);

    options.ralt = ralt;
  }

  rocksdb::DB *db;
  if (work_options.load) {
    std::cerr << "Emptying directories\n";
    empty_directory(db_path);
    for (auto path : options.db_paths) {
      empty_directory(path.path);
    }
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

  AutoTuner *autotuner = nullptr;
  if (vm.count("enable_auto_tuning") && ralt) {
    assert(first_level_in_sd > 0);
    size_t last_level_in_fd = first_level_in_sd - 1;
    uint64_t last_level_in_fd_size = level_size_path_id[last_level_in_fd].first;
    uint64_t fd_size = options.db_paths[0].target_size;
    autotuner = new AutoTuner(*db, first_level_in_sd, fd_size / 20,
                              last_level_in_fd_size * 0.8, fd_size / 20);
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
  delete ralt;

  return 0;
}
