#include <sys/resource.h>

#include "test.hpp"

typedef uint16_t field_size_t;

std::vector<std::filesystem::path> decode_db_paths(std::string db_paths) {
  std::istringstream in(db_paths);
  std::vector<std::filesystem::path> ret;
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
    ret.emplace_back(std::move(path));
    rusty_assert_eq(in.get(), '}', "Invalid db_paths");
    c = static_cast<char>(in.get());
    if (c != ',') break;
    rusty_assert_eq(in.get(), '{', "Invalid db_paths");
  }
  rusty_assert_eq(c, '}', "Invalid db_paths");
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

void bg_stat_printer(WorkOptions *work_options,
                     std::atomic<bool> *should_stop) {
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

  std::ofstream timers_out(db_path / "timers");
  timers_out << "Timestamp(ns) put-cpu-nanos "
                "get-cpu-nanos delete-cpu-nanos\n";

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
    work_options.num_keys = (work_options.ycsb_gen_options.record_count + 
                            work_options.ycsb_gen_options.operation_count 
                            * work_options.ycsb_gen_options.insert_proportion) * 1.1;
    num_keys = work_options.num_keys;
  } else {
    rusty_assert(vm.count("workload_file") == 0,
                 "workload_file only works with built-in generator!");
    rusty_assert(vm.count("export_key_only_trace") == 0,
                 "export_key_only_trace only works with built-in generator!");
    work_options.ycsb_gen_options = YCSBGen::YCSBGeneratorOptions();
    work_options.num_keys = num_keys;
  }

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

  leveldb::DB *db;
  if (work_options.load) {
    std::cerr << "Emptying directories\n";
    empty_directory(db_path);
    for (auto &path : db_paths) {
      empty_directory(path);
    }
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

  std::atomic<uint64_t> progress(0);
  std::atomic<uint64_t> progress_get(0);

  work_options.db = db;
  work_options.switches = switches;
  work_options.db_path = db_path;
  work_options.progress = &progress;
  work_options.progress_get = &progress_get;
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

  std::atomic<bool> should_stop(false);
  std::thread stat_printer(bg_stat_printer, &work_options, &should_stop);

  Tester tester(work_options);

  auto stats_print_func = [&](std::ostream &log) {
    log << "Timestamp: " << timestamp_ns() << "\n";
    std::string leveldb_stats;
    db->GetProperty("leveldb.stats", &leveldb_stats);
    log << "LevelDB stats: " << leveldb_stats << "\n";

    print_timers(log);

    db->ReportMigrationStats(log);

    /* Operation counts*/
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
    *info_json.lock() << "}" << std::endl;
  }

  should_stop.store(true, std::memory_order_relaxed);

  stats_print_func(std::cerr);
  stat_printer.join();
  period_print_thread.join();
  delete db;

  return 0;
}
