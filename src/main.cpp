#include "timers.h"

#include <boost/program_options/errors.hpp>
#include <boost/program_options/options_description.hpp>
#include <boost/program_options/parsers.hpp>
#include <boost/program_options/variables_map.hpp>
#include <iostream>
#include <filesystem>
#include <fstream>
#include <set>
#include <thread>
#include <chrono>

#include "rocksdb/db.h"
#include "rocksdb/filter_policy.h"
#include "rocksdb/statistics.h"
#include "rocksdb/table.h"

#ifndef crash_if
#define crash_if(cond, msg) do { \
	if (cond) { \
		fprintf(stderr, "crash_if: %s:%u: %s: Crashes due to %s: %s", \
			__FILE__, __LINE__, __func__, #cond, msg); \
		abort(); \
	} \
} while (0)
#endif

std::vector<rocksdb::DbPath>
decode_db_paths(std::string db_paths) {
	std::istringstream in(db_paths);
	std::vector<rocksdb::DbPath> ret;
	crash_if(in.get() != '{', "Invalid db_paths");
	char c = static_cast<char>(in.get());
	if (c == '}')
		return ret;
	crash_if(c != '{', "Invalid db_paths");
	while (1) {
		std::string path;
		size_t size;
		if (in.peek() == '"') {
			in >> std::quoted(path);
			crash_if(in.get() != ',', "Invalid db_paths");
		} else {
			while ((c = static_cast<char>(in.get())) != ',')
				path.push_back(c);
		}
		in >> size;
		ret.emplace_back(std::move(path), size);
		crash_if(in.get() != '}', "Invalid db_paths");
		c = static_cast<char>(in.get());
		if (c != ',')
			break;
		crash_if(in.get() != '{', "Invalid db_paths");
	}
	crash_if(c != '}', "Invalid db_paths");
	return ret;
}

int MaxBytesMultiplerAdditional(const rocksdb::Options& options, int level) {
	if (level >= static_cast<int>(options.max_bytes_for_level_multiplier_additional.size())) {
		return 1;
	}
	return options.max_bytes_for_level_multiplier_additional[level];
}

// Return the first level in the last path
int predict_level_assignment(const rocksdb::Options& options) {
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
			std::cerr << level << ' ' << options.db_paths[p].path << ' ' <<
				level_size << std::endl;
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
				level_size = static_cast<uint64_t>(
					static_cast<double>(level_size) *
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
	std::cerr << level << "+ " << options.db_paths[p].path << ' ' << level_size << std::endl;
	return level;
}

void empty_directory(std::filesystem::path dir_path) {
	for (auto& path : std::filesystem::directory_iterator(dir_path)) {
		std::filesystem::remove_all(path);
	}
}

bool is_empty_directory(std::string dir_path) {
	auto it = std::filesystem::directory_iterator(dir_path);
	return it == std::filesystem::end(it);
}

enum class TimerType : size_t {
	kInsert,
	kRead,
	kUpdate,
	kPut,
	kGet,
	kInputOperation,
	kInputInsert,
	kInputRead,
	kInputUpdate,
	kOutput,
	kSerialize,
	kDeserialize,
	kEnd,
};
const char *timer_names[] = {
	"Insert",
	"Read",
	"Update",
	"Put",
	"Get",
	"InputOperation",
	"InputInsert",
	"InputRead",
	"InputUpdate",
	"Output",
	"Serialize",
	"Deserialize",
};
TypedTimers<TimerType> timers;

int work_plain(rocksdb::DB *db, std::istream& in, std::ostream& ans_out) {
	while (1) {
		std::string op;
		in >> op;
		if (!in) {
			break;
		}
		if (op == "INSERT") {
			std::string key, value;
			in >> key >> value;
			rocksdb::Slice key_slice(key);
			rocksdb::Slice value_slice(value);
			auto s = db->Put(rocksdb::WriteOptions(), key_slice, value_slice);
			if (!s.ok()) {
				std::cerr << "INSERT failed with error: " << s.ToString() << std::endl;
				return -1;
			}
		} else if (op == "READ") {
			std::string key;
			in >> key;
			rocksdb::Slice key_slice(key);
			std::string value;
			auto s = db->Get(rocksdb::ReadOptions(), key_slice, &value);
			if (!s.ok()) {
				std::cerr << "GET failed with error: " << s.ToString() << std::endl;
				return -1;
			}
			ans_out << value << '\n';
		} else if (op == "UPDATE") {
			std::cerr << "UPDATE in plain format is not supported\n";
			return -1;
		} else {
			std::cerr << "Ignore line: " << op;
			std::getline(in, op); // Skip the rest of the line
			std::cerr << op << std::endl;
		}
	}
	return 0;
}

void handle_table_name(std::istream& in) {
	std::string table;
	in >> table;
	crash_if(table != "usertable", "Column families not supported yet.");
}

std::vector<std::pair<std::vector<char>, std::vector<char> > >
read_field_values(std::istream& in) {
	std::vector<std::pair<std::vector<char>, std::vector<char> > > ret;
	char c;
	do {
		c = static_cast<char>(in.get());
	} while (isspace(c));
	crash_if(c != '[', "Invalid KV trace!");
	crash_if(in.get() != ' ', "Invalid KV trace!");
	while (in.peek() != ']') {
		constexpr size_t vallen = 100;
		std::vector<char> field;
		std::vector<char> value(vallen);
		while ((c = static_cast<char>(in.get())) != '=') {
			field.push_back(c);
		}
		crash_if(!in.read(value.data(), vallen), "Invalid KV trace!");
		crash_if(in.get() != ' ', "Invalid KV trace!");
		ret.emplace_back(std::move(field), std::move(value));
	}
	in.get(); // ]
	return ret;
}

template <typename T>
void serialize_field_values(std::ostream& out, const T& fvs) {
	auto start_time = Timers::Start();
	for (const auto& fv : fvs) {
		size_t len = fv.first.size();
		out.write((char *)&len, sizeof(len));
		out.write(fv.first.data(), len);
		len = fv.second.size();
		out.write((char *)&len, sizeof(len));
		out.write(fv.second.data(), len);
	}
	timers.Stop(TimerType::kSerialize, start_time);
}

std::set<std::string> read_fields(std::istream& in) {
	char c;
	do {
		c = static_cast<char>(in.get());
	} while (isspace(c));
	crash_if(c != '[', "Invalid KV trace!");
	std::string s;
	std::getline(in, s);
	crash_if(s != " <all fields>]",
		"Reading specific fields is not supported yet.");
	return std::set<std::string>();
}

std::vector<char> read_len_bytes(std::istream& in) {
	size_t len;
	if (!in.read((char *)&len, sizeof(len))) {
		return std::vector<char>();
	}
	std::vector<char> bytes(len);
	crash_if(!in.read(bytes.data(), len), "Invalid KV trace!");
	return bytes;
}

std::map<std::vector<char>, std::vector<char> >
deserialize_values(std::istream& in,
		const std::set<std::string>& fields) {
	auto start_time = Timers::Start();
	crash_if(!fields.empty(), "Getting specific fields is not supported yet.");
	std::map<std::vector<char>, std::vector<char> > result;
	while (1) {
		auto field = read_len_bytes(in);
		if (!in) {
			break;
		}
		auto value = read_len_bytes(in);
		crash_if(!in, "Invalid KV trace!");
		crash_if(result.insert(std::make_pair(field, value)).second == false,
			"Duplicate field!");
	}
	timers.Stop(TimerType::kDeserialize, start_time);
	return result;
}

int work_ycsb(rocksdb::DB *db, std::istream& in, std::ostream& ans_out) {
	while (1) {
		std::string op;
		auto input_op_start =  Timers::Start();
		in >> op;
		timers.Stop(TimerType::kInputOperation, input_op_start);
		if (!in) {
			break;
		}
		if (op == "INSERT") {
			auto input_start = Timers::Start();
			handle_table_name(in);
			std::string key;
			in >> key;
			rocksdb::Slice key_slice(key);
			auto field_values = read_field_values(in);
			timers.Stop(TimerType::kInputInsert, input_start);

			auto insert_start = Timers::Start();
			std::ostringstream value_out;
			serialize_field_values(value_out, field_values);
			// TODO: Avoid the copy
			std::string value = value_out.str();
			auto value_slice =
				rocksdb::Slice(value.c_str(), value.size());
			auto put_start = Timers::Start();
			auto s = db->Put(rocksdb::WriteOptions(), key_slice, value_slice);
			timers.Stop(TimerType::kPut, put_start);
			if (!s.ok()) {
				std::cerr << "INSERT failed with error: " << s.ToString() << std::endl;
				return -1;
			}
			timers.Stop(TimerType::kInsert, insert_start);
		} else if (op == "READ") {
			auto input_start = Timers::Start();
			handle_table_name(in);
			std::string key;
			in >> key;
			rocksdb::Slice key_slice(key);
			auto fields = read_fields(in);
			timers.Stop(TimerType::kInputRead, input_start);

			auto read_start = Timers::Start();
			std::string value;
			auto get_start = Timers::Start();
			auto s = db->Get(rocksdb::ReadOptions(), key_slice, &value);
			timers.Stop(TimerType::kGet, get_start);
			if (!s.ok()) {
				std::cerr << "GET failed with error: " << s.ToString() << std::endl;
				return -1;
			}
			std::istringstream value_in(value);
			auto result = deserialize_values(value_in, fields);
			timers.Stop(TimerType::kRead, read_start);

			auto output_start = Timers::Start();
			ans_out << "[ ";
			for (const auto& field_value : result) {
				ans_out.write(field_value.first.data(),
					static_cast<std::streamsize>(field_value.first.size()));
				ans_out << ' ';
				ans_out.write(field_value.second.data(),
					static_cast<std::streamsize>(field_value.second.size()));
				ans_out << ' ';
			}
			ans_out << "]\n";
			timers.Stop(TimerType::kOutput, output_start);
		} else if (op == "UPDATE") {
			auto input_start = Timers::Start();
			handle_table_name(in);
			std::string key;
			in >> key;
			rocksdb::Slice key_slice(key);
			auto updates = read_field_values(in);
			timers.Stop(TimerType::kInputUpdate, input_start);

			auto update_start = Timers::Start();
			std::string value;
			auto get_start = Timers::Start();
			auto s = db->Get(rocksdb::ReadOptions(), key_slice, &value);
			timers.Stop(TimerType::kGet, get_start);
			if (!s.ok()) {
				std::cerr << "GET failed with error: " << s.ToString() << std::endl;
				return -1;
			}
			std::istringstream value_in(value);
			auto values = deserialize_values(value_in, std::set<std::string>());
			for (const auto& update : updates) {
				values[update.first] = update.second;
			}
			std::ostringstream value_out;
			serialize_field_values(value_out, values);
			value = value_out.str();
			auto value_slice =
				rocksdb::Slice(value.c_str(), value.size());
			auto put_start = Timers::Start();
			s = db->Put(rocksdb::WriteOptions(), key_slice, value_slice);
			timers.Stop(TimerType::kPut, put_start);
			if (!s.ok()) {
				std::cerr << "UPDATE failed with error: " << s.ToString() << std::endl;
				return -1;
			}
			timers.Stop(TimerType::kUpdate, update_start);
		}
		else {
			std::cerr << "Ignore line: " << op;
			std::getline(in, op); // Skip the rest of the line
			std::cerr << op << std::endl;
		}
	}
	return 0;
}

bool has_background_work(rocksdb::DB *db) {
	uint64_t flush_pending;
	uint64_t compaction_pending;
	uint64_t flush_running;
	uint64_t compaction_running;
	bool ok =
		db->GetIntProperty(
			rocksdb::Slice("rocksdb.mem-table-flush-pending"), &flush_pending);
	// assert(ok);
	crash_if(!ok, "");
	ok = db->GetIntProperty(
			rocksdb::Slice("rocksdb.compaction-pending"), &compaction_pending);
	// assert(ok);
	crash_if(!ok, "");
	ok = db->GetIntProperty(
			rocksdb::Slice("rocksdb.num-running-flushes"), &flush_running);
	// assert(ok);
	crash_if(!ok, "");
	ok = db->GetIntProperty(
			rocksdb::Slice("rocksdb.num-running-compactions"),
			&compaction_running);
	// assert(ok);
	crash_if(!ok, "");
	return flush_pending || compaction_pending || flush_running ||
		compaction_running;
}

void wait_for_background_work(rocksdb::DB *db) {
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
			// std::cerr << "There is no background work detected for more than 2 seconds. Exiting...\n";
			break;
		}
	}
}

int main(int argc, char **argv) {
	rocksdb::Options options;

	namespace po = boost::program_options;
	po::options_description desc("Available options");
	std::string format;
	std::string arg_db_path;
	std::string arg_db_paths;
	size_t cache_size;
	desc.add_options()
		("help", "Print help message")
		("cleanup,c", "Empty the directories first.")
		(
			"format,f", po::value<std::string>(&format)->default_value("ycsb"),
			"Trace format: plain/ycsb"
		) (
			"use_direct_reads",
			po::value<bool>(&options.use_direct_reads)->default_value(true), ""
		) (
			"db_path", po::value<std::string>(&arg_db_path)->required(),
			"Path to database"
		) (
			"db_paths", po::value<std::string>(&arg_db_paths)->required(),
			"For example: \"{{/tmp/sd,100000000},{/tmp/cd,1000000000}}\""
		) (
			"cache_size",
			po::value<size_t>(&cache_size)->default_value(8 << 20),
			"Capacity of LRU block cache in bytes. Default: 8MiB"
		);
	po::variables_map vm;
	po::store(po::parse_command_line(argc, argv, desc), vm);
	if (vm.count("help")) {
		std::cerr << desc << std::endl;
		return 1;
	}
	po::notify(vm);

	std::filesystem::path db_path(arg_db_path);
	options.db_paths = decode_db_paths(arg_db_paths);
	options.statistics = rocksdb::CreateDBStatistics();

	rocksdb::BlockBasedTableOptions table_options;
	table_options.block_cache = rocksdb::NewLRUCache(cache_size);
	table_options.filter_policy.reset(rocksdb::NewBloomFilterPolicy(10, false));
	options.table_factory.reset(
		rocksdb::NewBlockBasedTableFactory(table_options));

	if (vm.count("cleanup")) {
		std::cerr << "Emptying directories\n";
		empty_directory(db_path);
		for (auto path : options.db_paths) {
			empty_directory(path.path);
		}
	}
	predict_level_assignment(options);

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

	int ret;
	auto start = std::chrono::steady_clock::now();
	if (format == "plain") {
		ret = work_plain(db, std::cin, std::cout);
	} else if (format == "ycsb") {
		ret = work_ycsb(db, std::cin, std::cout);
	} else {
		std::cerr << "Unrecognized format: " << format << std::endl;
		ret = -1;
	}
	auto end = std::chrono::steady_clock::now();
	std::cerr << (double)std::chrono::duration_cast<std::chrono::nanoseconds>(
			end - start).count() / 1e9 << " second(s) for work\n";

	start = std::chrono::steady_clock::now();
	wait_for_background_work(db);
	end = std::chrono::steady_clock::now();
	std::cerr << (double)std::chrono::duration_cast<std::chrono::nanoseconds>(
			end - start).count() / 1e9 <<
		" second(s) waiting for background work\n";

	std::cerr << "rocksdb.block.cache.data.miss: "
		<< options.statistics->getTickerCount(rocksdb::BLOCK_CACHE_DATA_MISS)
		<< std::endl;
	std::cerr << "rocksdb.block.cache.data.hit: "
		<< options.statistics->getTickerCount(rocksdb::BLOCK_CACHE_DATA_HIT)
		<< std::endl;
	std::cerr << "rocksdb.bloom.filter.useful: "
		<< options.statistics->getTickerCount(rocksdb::BLOOM_FILTER_USEFUL)
		<< std::endl;
	std::cerr << "rocksdb.bloom.filter.full.positive: "
		<< options.statistics->getTickerCount(
			rocksdb::BLOOM_FILTER_FULL_POSITIVE)
		<< std::endl;
	std::cerr << "rocksdb.memtable.hit: " <<
		options.statistics->getTickerCount(rocksdb::MEMTABLE_HIT) << std::endl;
	std::cerr << "rocksdb.l0.hit: " <<
		options.statistics->getTickerCount(rocksdb::GET_HIT_L0) << std::endl;
	std::cerr << "rocksdb.l1.hit: " <<
		options.statistics->getTickerCount(rocksdb::GET_HIT_L1) << std::endl;
	std::cerr << "rocksdb.rocksdb.l2andup.hit: " <<
		options.statistics->getTickerCount(rocksdb::GET_HIT_L2_AND_UP) <<
		std::endl;

	std::string rocksdb_stats;
	crash_if(!db->GetProperty("rocksdb.stats", &rocksdb_stats), "");
	std::ofstream(db_path / "rocksdb-stats.txt") << rocksdb_stats;

	auto timers_status = timers.Collect();
	for (size_t i = 0; i < static_cast<size_t>(TimerType::kEnd); ++i) {
		std::cerr << timer_names[i] << ": count " << timers_status[i].count <<
			", total " << timers_status[i].nsec << "ns\n";
	}
	std::cerr << "In summary: [\n";
	Timers::Status input_time =
		timers_status[static_cast<size_t>(TimerType::kInputOperation)] +
			timers_status[static_cast<size_t>(TimerType::kInputInsert)] +
			timers_status[static_cast<size_t>(TimerType::kInputRead)] +
			timers_status[static_cast<size_t>(TimerType::kInputUpdate)];
	std::cerr << "\tInput: count " << input_time.count << ", total " <<
		input_time.nsec << "ns\n";
	std::cerr << "]\n";

	delete db;

	return ret;
}
