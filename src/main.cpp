#include "test.hpp"

using boost::fibers::buffered_channel;

typedef uint16_t field_size_t;

std::vector<std::filesystem::path> decode_db_paths_to_filepaths(std::string db_paths) {
  std::istringstream in(db_paths);
  std::vector<std::filesystem::path> ret;
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
    ret.emplace_back(std::move(path));
    rusty_assert(in.get() == '}', "Invalid db_paths");
    c = static_cast<char>(in.get());
    if (c != ',') break;
    rusty_assert(in.get() == '{', "Invalid db_paths");
  }
  rusty_assert(c == '}', "Invalid db_paths");
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

bool has_background_work(leveldb::DB *db) {
  return false;
}

void wait_for_background_work(leveldb::DB *db) {
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
  while (!should_stop->load(std::memory_order_relaxed)) {
    auto timestamp = timestamp_ns();
    auto value = progress->load(std::memory_order_relaxed);
    progress_out << timestamp << ' ' << value << std::endl;
    std::this_thread::sleep_for(std::chrono::seconds(1));
  }
}

int main(int argc, char **argv) {
  leveldb::Options options;

  namespace po = boost::program_options;
  po::options_description desc("Available options");
  std::string format;
  std::string arg_db_path;
  std::string arg_db_paths;
  size_t cache_size;
  std::string arg_switches;
  size_t num_threads;
  size_t num_keys;
  size_t num_load_ops;
  double optane_threshold;
  size_t num_write_keys;
  desc.add_options()("help", "Print help message");
  desc.add_options()("cleanup,c", "Empty the directories first.");
  desc.add_options()("enable_fast_process", "Enable fast processing method.");
  desc.add_options()("format,f",
                     po::value<std::string>(&format)->default_value("ycsb"),
                     "Trace format: plain/ycsb");
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
  desc.add_options()("num_keys",
                     po::value<size_t>(&num_keys)->required(),
                     "The number of keys.\n");
  desc.add_options()("migrations_logging",
                      po::value<bool>(&options.migration_logging)->required(), "Option migrations_logging");
  desc.add_options()("read_logging",
                      po::value<bool>(&options.read_logging)->required(), "Option read_logging");
  desc.add_options()("migration_policy",
                      po::value<int>(&options.migration_policy)->required(), "Option migration_policy");
  desc.add_options()("migration_metric",
                      po::value<int>(&options.migration_metric)->required(), "Option migration_metric");
  desc.add_options()("migration_rand_range_num",
                      po::value<int>(&options.migration_rand_range_num)->required(), "Option migration_rand_range_num");
  desc.add_options()("migration_rand_range_size",
                      po::value<int>(&options.migration_rand_range_size)->required(), "Option migration_rand_range_size");
  desc.add_options()("num_load_ops", po::value<size_t>(&num_load_ops)->required(), "Number of operations in loading phase.");
  desc.add_options()("optane_threshold", po::value<double>(&optane_threshold)->default_value(0.15), "Optane threshold.");
  desc.add_options()("slab_dir", po::value<std::string>(&options.slab_dir)->required(), "Directory of slabs.");
  desc.add_options()("pop_cache_size", po::value<uint32_t>(&options.popCacheSize)->required(), "size of popularity cache.");
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

  //PrismDB
  options.env = leveldb::Env::Default();
  options.block_cache = leveldb::NewLRUCache(cache_size);
  options.filter_policy = leveldb::NewBloomFilterPolicy(10);
  options.numKeys = num_keys;
  options.optaneThreshold = optane_threshold;
  options.numWriteKeys = num_keys;

  std::filesystem::path db_path(arg_db_path);
  auto db_paths = decode_db_paths_to_filepaths(arg_db_paths);
  // options.db_paths = decode_db_paths(arg_db_paths);
  // options.statistics = leveldb::CreateDBStatistics();

  // leveldb::BlockBasedTableOptions table_options;
  // table_options.block_cache = leveldb::NewLRUCache(cache_size);
  // table_options.filter_policy.reset(leveldb::NewBloomFilterPolicy(10, false));
  // options.table_factory.reset(
  //     leveldb::NewBlockBasedTableFactory(table_options));

  if (vm.count("cleanup")) {
    std::cerr << "Emptying directories\n";
    empty_directory(db_path);
    for (auto& path : db_paths) {
      empty_directory(path);
    }
    options.create_if_missing = true;
  }

  leveldb::DB *db;
  auto s = leveldb::DB::Open(options, db_path.string(), &db);
  if (!s.ok()) {
    std::cerr << s.ToString() << std::endl;
    return -1;
  }

  std::atomic<bool> should_stop(false);
  std::atomic<size_t> progress(0);
  std::thread stat_printer(bg_stat_printer, db_path, &should_stop, &progress);

  std::string pid = std::to_string(getpid());
  std::string cmd = "pidstat -p " + pid +
                    " -Hu 1 | "
                    "awk '{if(NR>3){print $1,$8; fflush(stdout)}}' > " +
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
  work_option.num_keys = num_keys;
  work_option.num_load_ops = num_load_ops;
  work_option.format_type = format == "ycsb" ? FormatType::YCSB : FormatType::Plain;
  Tester tester(work_option);

  auto stats_print_func = [&] (std::ostream& log) {
    log << "Timestamp: " << timestamp_ns() << "\n";
    std::string leveldb_stats;
    db->GetProperty("leveldb.stats", &leveldb_stats);
    log << "LevelDB stats: " << leveldb_stats << "\n";
    log << "Timers: \n";
    std::vector<counter_timer::CountTime> timers_status;
    const auto &ts = timers.timers();
    size_t num_types = ts.len();
    for (size_t i = 0; i < num_types; ++i) {
      const auto &timer = ts.timer(i);
      uint64_t count = timer.count();
      rusty::time::Duration time = timer.time();
      timers_status.push_back(counter_timer::CountTime{count, time});
      std::cerr << timer_names[i] << ": count " << count << ", total "
                << time.as_nanos() << "ns\n";
    }
    log << "end===\n";
    db->ReportMigrationStats(log);
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
  stat_printer.join();
  period_print_thread.join();
  delete db;

  return 0;
}
