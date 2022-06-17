#include <iostream>
#include <filesystem>
#include <fstream>
#include <random>
#include <set>
#include <thread>
#include <chrono>

#include "rocksdb/db.h"

#define crash_if(cond, msg) do { \
	if (cond) { \
		fprintf(stderr, "crash_if: %s:%u: %s: Crashes due to %s: %s", \
			__FILE__, __LINE__, __func__, #cond, msg); \
		abort(); \
	} \
} while (0)

std::vector<rocksdb::DbPath>
decode_db_paths(std::string db_paths) {
	std::istringstream in(db_paths);
	std::vector<rocksdb::DbPath> ret;
	crash_if(in.get() != '{', "Invalid db_paths");
	char c = in.get();
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
			while ((c = in.get()) != ',')
				path.push_back(c);
		}
		in >> size;
		// std::cout << path << "," << size << std::endl;
		ret.emplace_back(std::move(path), size);
		crash_if(in.get() != '}', "Invalid db_paths");
		c = in.get();
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

	std::cout << "Predicted level assignment:\n";

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
			std::cout << level << ' ' << options.db_paths[p].path << ' ' <<
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
					level_size * options.max_bytes_for_level_multiplier);
			} else {
				level_size = static_cast<uint64_t>(
					level_size * options.max_bytes_for_level_multiplier *
						MaxBytesMultiplerAdditional(options, cur_level));
			}
		}
		cur_level++;
	}
	std::cout << level << "+ " << options.db_paths[p].path << ' ' << level_size << std::endl;
	return level;
}

void empty_directory(std::string dir_path) {
	for (auto& path : std::filesystem::directory_iterator(dir_path)) {
		std::filesystem::remove_all(path);
	}
}

bool is_empty_directory(std::string dir_path) {
	auto it = std::filesystem::directory_iterator(dir_path);
	return it == std::filesystem::end(it);
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
		c = in.get();
	} while (isspace(c));
	crash_if(c != '[', "Invalid KV trace!");
	crash_if(in.get() != ' ', "Invalid KV trace!");
	while (in.peek() != ']') {
		constexpr size_t vallen = 100;
		std::vector<char> field;
		std::vector<char> value(vallen);
		while ((c = in.get()) != '=') {
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
	for (const auto& fv : fvs) {
		size_t len = fv.first.size();
		out.write((char *)&len, sizeof(len));
		out.write(fv.first.data(), len);
		len = fv.second.size();
		out.write((char *)&len, sizeof(len));
		out.write(fv.second.data(), len);
	}
}

std::set<std::string> read_fields(std::istream& in) {
	char c;
	do {
		c = in.get();
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
	return result;
}

int work(rocksdb::DB *db, std::istream& in, std::ostream& ans_out) {
	while (1) {
		std::string op;
		in >> op;
		if (!in) {
			break;
		}
		if (op == "INSERT") {
			handle_table_name(in);
			std::string key;
			in >> key;
			rocksdb::Slice key_slice(key);
			std::ostringstream value_out;
			serialize_field_values(value_out, read_field_values(in));
			// TODO: Avoid the copy
			std::string value = value_out.str();
			auto value_slice =
				rocksdb::Slice(value.c_str(), value.size());
			auto s = db->Put(rocksdb::WriteOptions(), key_slice, value_slice);
			if (!s.ok()) {
				std::cout << "INSERT failed with error: " << s.ToString() << std::endl;
				return -1;
			}
		} else if (op == "READ") {
			handle_table_name(in);
			std::string key;
			in >> key;
			rocksdb::Slice key_slice(key);
			auto fields = read_fields(in);
			std::string value;
			auto s = db->Get(rocksdb::ReadOptions(), key_slice, &value);
			if (!s.ok()) {
				std::cout << "GET failed with error: " << s.ToString() << std::endl;
				return -1;
			}
			std::istringstream value_in(value);
			auto result = deserialize_values(value_in, fields);
			ans_out << "[ ";
			for (const auto& field_value : result) {
				ans_out.write(field_value.first.data(), field_value.first.size());
				ans_out << ' ';
				ans_out.write(field_value.second.data(), field_value.second.size());
				ans_out << ' ';
			}
			ans_out << "]\n";
		} else if (op == "UPDATE") {
			handle_table_name(in);
			std::string key;
			in >> key;
			rocksdb::Slice key_slice(key);
			auto updates = read_field_values(in);
			std::string value;
			auto s = db->Get(rocksdb::ReadOptions(), key_slice, &value);
			if (!s.ok()) {
				std::cout << "GET failed with error: " << s.ToString() << std::endl;
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
			s = db->Put(rocksdb::WriteOptions(), key_slice, value_slice);
			if (!s.ok()) {
				std::cout << "UPDATE failed with error: " << s.ToString() << std::endl;
				return -1;
			}
		}
		else {
			std::cout << "Ignore line: " << op;
			std::getline(in, op); // Skip the rest of the line
			std::cout << op << std::endl;
		}
	}
	return 0;
}

class RouterTrivial : public rocksdb::CompactionRouter {
	void Access(int, const rocksdb::Slice&, size_t) override {}
	rocksdb::CompactionRouter::Decision
	Route(int, const rocksdb::Slice&) override {
		return rocksdb::CompactionRouter::Decision::kNextLevel;
	}
	const char *Name() const override {
		return "RouterTrivial";
	}
};

class RouterProb : public rocksdb::CompactionRouter {
public:
	RouterProb(float prob_to_next, unsigned int seed)
	  : prob_to_next_(prob_to_next),
		gen_(seed),
		dis_(0, 1) {}
	void Access(int, const rocksdb::Slice&, size_t) override {}
	rocksdb::CompactionRouter::Decision
	Route(int, const rocksdb::Slice &) override {
		if (dis_(gen_) >= prob_to_next_)
			return rocksdb::CompactionRouter::Decision::kNextLevel;
		else
			return rocksdb::CompactionRouter::Decision::kCurrentLevel;
	}
	const char *Name() const override {
		return "RouterProb";
	}
private:
	float prob_to_next_;
	std::mt19937 gen_;
	std::uniform_real_distribution<float> dis_;
};

extern void *VisCntsOpen(const char *path, double delta, bool createIfMissing);
extern int VisCntsAccess(void *ac, const char *key, size_t klen, size_t vlen);
extern bool VisCntsIsHot(void *ac, const char *key, size_t klen);
extern int VisCntsClose(void *ac);

class RouterVisCnts : public rocksdb::CompactionRouter {
public:
	RouterVisCnts(int target_level, const char *path, double delta,
			bool create_if_missing)
		:	ac_(VisCntsOpen(path, delta, create_if_missing)),
			target_level_(target_level),
			accessed_(0),
			retained_(0),
			not_retained_(0) {}
	~RouterVisCnts() {
		VisCntsClose(ac_);
	}
	void Access(int level, const rocksdb::Slice &key, size_t vlen)
			override {
		if (level < target_level_)
			return;
		accessed_.fetch_add(1, std::memory_order_relaxed);
		VisCntsAccess(ac_, key.data(), key.size(), vlen);
	}
	rocksdb::CompactionRouter::Decision
	Route(int level, const rocksdb::Slice& key) override {
		if (level != target_level_)
			return rocksdb::CompactionRouter::Decision::kNextLevel;
		if (VisCntsIsHot(ac_, key.data(), key.size())) {
			retained_.fetch_add(1, std::memory_order_relaxed);
			return rocksdb::CompactionRouter::Decision::kCurrentLevel;
		} else {
			not_retained_.fetch_add(1, std::memory_order_relaxed);
			return rocksdb::CompactionRouter::Decision::kNextLevel;
		}
	}
	const char *Name() const override {
		return "RouterVisCnts";
	}
	size_t accessed() const {
		return accessed_.load(std::memory_order_relaxed);
	}
	size_t retained() const {
		return retained_.load(std::memory_order_relaxed);
	}
	size_t not_retained() const {
		return not_retained_.load(std::memory_order_relaxed);
	}
private:
	void *ac_;
	int target_level_;
	std::atomic<size_t> accessed_;
	std::atomic<size_t> retained_;
	std::atomic<size_t> not_retained_;
};

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
			// std::cout << "There is no background work detected for more than 2 seconds. Exiting...\n";
			break;
		}
	}
}

int main(int argc, char **argv) {
	if (argc != 9) {
		std::cout << argc << std::endl;
		std::cout << "Usage:\n";
		std::cout << "Arg 1: Whether to empty the directories.\n";
		std::cout << "\t1: Empty the directories first.\n";
		std::cout << "\t0: Leave the directories as they are.\n";
		std::cout << "Arg 2: Use O_DIRECT for user and compaction reads?\n";
		std::cout << "\t1: Yes\n";
		std::cout << "\t0: No\n";
		std::cout << "Arg 3: Path to database\n";
		std::cout << "Arg 4: db_paths, for example: "
			"\"{{/tmp/sd,100000000},{/tmp/cd,1000000000}}\"\n";
		std::cout << "Arg 5: Path to KV operation trace file\n";
		std::cout << "Arg 6: Path to save output\n";
		std::cout << "Arg 7: Path to VisCnts\n";
		std::cout << "Arg 8: Delta in bytes\n";
		return -1;
	}
	rocksdb::Options options;

	bool empty_directories_first = (argv[1][0] == '1');
	options.use_direct_reads = (argv[2][0] == '1');
	std::string db_path = std::string(argv[3]);
	std::string db_paths(argv[4]);
	std::string kvops_path = std::string(argv[5]);
	std::string ans_out_path = std::string(argv[6]);
	const char *viscnts_path = argv[7];
	double delta = atof(argv[8]);
	options.write_buffer_size = 1 << 20;
	options.target_file_size_base = 1 << 20;
	options.max_bytes_for_level_base = 4 * options.target_file_size_base;

	options.db_paths = decode_db_paths(db_paths);

	if (empty_directories_first) {
		std::cout << "Emptying directories\n";
		empty_directory(db_path);
		for (auto path : options.db_paths) {
			empty_directory(path.path);
		}
		empty_directory(viscnts_path);
	}

	int first_cd_level = predict_level_assignment(options);

	// options.compaction_router = new RouterTrivial;
	// options.compaction_router = new RouterProb(0.5, 233);
	auto router =
		new RouterVisCnts(first_cd_level - 1, viscnts_path, delta, true);
	options.compaction_router = router;

	std::ifstream in(kvops_path);
	if (!in) {
		std::cout << "Fail to open " << kvops_path << std::endl;
		return -1;
	}

	std::ofstream ans_out(ans_out_path);
	if (!ans_out) {
		std::cout << "Fail to open " << ans_out_path << std::endl;
		return -1;
	}

	rocksdb::DB *db;
	auto s = rocksdb::DB::Open(options, db_path, &db);
	if (!s.ok()) {
		std::cout << "Creating database\n";
		options.create_if_missing = true;
		s = rocksdb::DB::Open(options, db_path, &db);
		if (!s.ok()) {
			std::cout << s.ToString() << std::endl;
			return -1;
		}
	}

	auto start = std::chrono::steady_clock::now();
	int ret = work(db, in, ans_out);
	auto end = std::chrono::steady_clock::now();
	std::cout << (double)std::chrono::duration_cast<std::chrono::nanoseconds>(
			end - start).count() / 1e9 << " second(s) for work\n";

	start = std::chrono::steady_clock::now();
	wait_for_background_work(db);
	end = std::chrono::steady_clock::now();
	std::cout << (double)std::chrono::duration_cast<std::chrono::nanoseconds>(
			end - start).count() / 1e9 <<
		" second(s) waiting for background work\n";

	std::cout << "Accessed: " << router->accessed() << std::endl;
	std::cout << "Retained: " << router->retained() << std::endl;
	std::cout << "Not retained: " << router->not_retained() << std::endl;

	delete db;
	delete router;

	return ret;
}
