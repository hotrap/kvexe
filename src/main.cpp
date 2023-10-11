#include "test.hpp"

typedef uint16_t field_size_t;

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
  auto mem_path = db_path / "mem";
  std::ofstream(mem_path) << "Timestamp(ns) RSS(KB)\n";

  while (!should_stop->load(std::memory_order_relaxed)) {
    auto timestamp = timestamp_ns();
    auto value = progress->load(std::memory_order_relaxed);
    progress_out << timestamp << ' ' << value << std::endl;

    std::ofstream(mem_path, std::ios_base::app) << timestamp << ' ';
    std::system(("ps -q " + std::to_string(getpid()) +
                 " -o rss | tail -n 1 >> " + mem_path.c_str())
                    .c_str());

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
  size_t cache_size;
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

  uint64_t switches;
  if (arg_switches == "none") {
    switches = 0;
  } else if (arg_switches == "all") {
    switches = 0x3;
  } else {
    std::istringstream in(std::move(arg_switches));
    in >> std::hex >> switches;
  }

  std::filesystem::path db_path(arg_db_path);
  options.db_paths = decode_db_paths(arg_db_paths);
  options.statistics = rocksdb::CreateDBStatistics();

  options.max_background_jobs = 4;

  // To slow down the execution to make time for compactions
  options.level0_file_num_compaction_trigger = 1;
  // options.level0_slowdown_writes_trigger = 2;
  // options.level0_stop_writes_trigger = 4;

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
  }
  options.max_bytes_for_level_multiplier_additional =
      std::vector<double>({1.0, 1.745});
  int first_level_in_cd = predict_level_assignment(options);
  std::ofstream(db_path / "first-level-in-cd")
      << first_level_in_cd << std::endl;

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
  std::thread stat_printer(bg_stat_printer, db_path, &should_stop, &progress);

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

  return 0;
}
