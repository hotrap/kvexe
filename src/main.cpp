#include "timers.h"

#include <iostream>
#include <filesystem>
#include <fstream>
#include <random>
#include <set>
#include <thread>
#include <queue>

#include "rocksdb/db.h"
#include "rcu_vector_bp.hpp"

#include "viscnts.h"

#ifndef crash_if
#define panic(...) do { \
    fprintf(stderr, "panic: %s:%u: %s:", \
        __FILE__, __LINE__, __func__); \
    fprintf(stderr, " " __VA_ARGS__);   \
    abort(); \
} while (0)
#define crash_if(cond, ...) do { \
    if (cond) { panic(__VA_ARGS__); }   \
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

void empty_directory(std::string dir_path) {
	for (auto& path : std::filesystem::directory_iterator(dir_path)) {
		std::filesystem::remove_all(path);
	}
}

bool is_empty_directory(std::string dir_path) {
	auto it = std::filesystem::directory_iterator(dir_path);
	return it == std::filesystem::end(it);
}

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

int work_ycsb(rocksdb::DB *db, std::istream& in, std::ostream& ans_out) {
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
				std::cerr << "INSERT failed with error: " << s.ToString() << std::endl;
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
				std::cerr << "GET failed with error: " << s.ToString() << std::endl;
				return -1;
			}
			std::istringstream value_in(value);
			auto result = deserialize_values(value_in, fields);
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
		} else if (op == "UPDATE") {
			handle_table_name(in);
			std::string key;
			in >> key;
			rocksdb::Slice key_slice(key);
			auto updates = read_field_values(in);
			std::string value;
			auto s = db->Get(rocksdb::ReadOptions(), key_slice, &value);
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
			s = db->Put(rocksdb::WriteOptions(), key_slice, value_slice);
			if (!s.ok()) {
				std::cerr << "UPDATE failed with error: " << s.ToString() << std::endl;
				return -1;
			}
		}
		else {
			std::cerr << "Ignore line: " << op;
			std::getline(in, op); // Skip the rest of the line
			std::cerr << op << std::endl;
		}
	}
	return 0;
}

enum class TimerType {
	kRangeHotSize,
	kEnd,
};
const char *timer_names[] = {
	"RangeHotSize",
};
TypedTimers<TimerType, timer_names> timers;

class RouterVisCnts : public rocksdb::CompactionRouter {
public:
	RouterVisCnts(const rocksdb::Comparator *ucmp, int target_level,
			const char *dir, double weight_sum_max, bool create_if_missing)
		:	ucmp_(ucmp),
			dir_(dir),
			create_if_missing_(create_if_missing),
			tier0_last_level_(target_level),
			weight_sum_max_(weight_sum_max),
			notify_weight_change_(2),
			new_iter_cnt_(0),
			hot_taken_(0) {}
	~RouterVisCnts() {
		size_t size = vcs_.size_locked();
		for (size_t i = 0; i < size; ++i) {
			delete vcs_.ref_locked(i);
		}
		size = accessed_.size_locked();
		for (size_t i = 0; i < size; ++i) {
			delete accessed_.ref_locked(i);
		}
	}
	const char *Name() const override {
		return "RouterVisCnts";
	}
	size_t Tier(int level) override {
		if (level <= tier0_last_level_) {
			return 0;
		} else {
			return 1;
		}
	}
	void AddHotness(size_t tier, const rocksdb::Slice& key, size_t vlen,
			double weight) override {
		add(add_hotness_cnts_, tier, (size_t)1);
		addHotness(tier, key, vlen, weight);
	}
	void Access(int level, const rocksdb::Slice& key, size_t vlen)
			override {
		if (level < tier0_last_level_)
			return;
		add(accessed_, (size_t)level, (size_t)1);

		size_t tier = Tier(level);
		if (vcs_.size() <= (size_t)tier) {
			vcs_.lock();
			while (vcs_.size_locked() <= (size_t)tier) {
				std::filesystem::path dir(dir_);
				auto path = dir / std::to_string(vcs_.size_locked());
				vcs_.push_back_locked(
					new VisCnts(ucmp_, path.c_str(), create_if_missing_,
						&notify_weight_change_));
			}
			vcs_.unlock();
		}
		addHotness(tier, key, vlen, 1);
	}
	void *NewIter(size_t tier) override {
		if (vcs_.size() <= tier)
			return NULL;
		new_iter_cnt_.fetch_add(1, std::memory_order_relaxed);
		VisCnts* vc = vcs_.read_copy(tier);
		return new VisCnts::Iter(vc);
	}
	// The returned pointer will stay valid until the next call to Seek or
	// NextHot with this iterator
	const rocksdb::HotRecInfo *Seek(void *iter, const rocksdb::Slice& key)
			override {
		if (iter == NULL)
			return NULL;
		auto it = (VisCnts::Iter*)iter;
		return it->Seek(key);
	}
	const rocksdb::HotRecInfo *NextHot(void *iter) override {
		if (iter == NULL)
			return NULL;
		auto it = (VisCnts::Iter*)iter;
		const rocksdb::HotRecInfo *ret = it->Next();
		if (ret) {
			hot_taken_.fetch_add(1, std::memory_order_relaxed);
		}
		return ret;
	}
	void DelIter(void *iter) override {
		if (iter == NULL)
			return;
		delete (VisCnts::Iter*)iter;
	}
	void DelRange(size_t tier, const rocksdb::Slice& smallest,
			const rocksdb::Slice& largest) override {
		if (vcs_.size() <= tier)
			return;
		vcs_.read_copy(tier)->RangeDel(smallest, largest);
	}
	size_t RangeHotSize(size_t tier, const rocksdb::Slice& smallest,
			const rocksdb::Slice& largest) override {
		if (vcs_.size() <= tier)
			return 0;
		auto start = timers.Start();
		size_t ret = vcs_.read_copy(tier)->RangeHotSize(smallest, largest);
		timers.Stop(TimerType::kRangeHotSize, start);
		return ret;
	}
	std::vector<size_t> accessed() {
		size_t size = accessed_.size();
		std::vector<size_t> ret(size);
		for (size_t i = 0; i < size; ++i) {
			auto val = accessed_.read_copy(i);
			ret[i] = val->load(std::memory_order_relaxed);
		}
		return ret;
	}
	size_t new_iter_cnt() {
		return new_iter_cnt_.load(std::memory_order_relaxed);
	}
	size_t hot_taken() {
		return hot_taken_.load(std::memory_order_relaxed);
	}
	std::vector<size_t> add_hotness_cnts() {
		size_t size = add_hotness_cnts_.size();
		std::vector<size_t> ret(size);
		for (size_t i = 0; i < size; ++i) {
			auto val = add_hotness_cnts_.read_copy(i);
			ret[i] = val->load(std::memory_order_relaxed);
		}
		return ret;
	}
	std::string sprint_viscnts() {
		std::stringstream out;
		size_t size = vcs_.size();
		out << "[";
		for (size_t i = 0; i < size; ++i) {
			void *iter = NewIter(i);
			if (iter == NULL)
				continue;
			out << "{" << "\"level\": " << i << ", \"hot\": [";
			auto it = (VisCnts::Iter*)iter;
			it->SeekToFirst();
			while (1) {
				auto info = it->Next();
				if (info == NULL)
					break;
				out << "{\"key\": \"" << info->slice.ToString() << '"' <<
					", \"count\": " << info->count <<
					", \"vlen\": " << info->vlen << "},";
			}
			DelIter(iter);
			out << "]},";
		}
		out << "]";
		return out.str();
	}
private:
	template <typename T>
	void prepare(rcu_vector_bp<std::atomic<T> *>& v, size_t i) {
		if (v.size() <= i) {
			v.lock();
			while (v.size_locked() <= i) {
				v.push_back_locked(new std::atomic<T>(0));
			}
			v.unlock();
		}
	}
	template <typename T>
	void add(rcu_vector_bp<std::atomic<T> *>& v, size_t i, T val) {
		prepare(v, i);
		v.read_copy(i)->fetch_add(val, std::memory_order_relaxed);
	}
	double weightSum() {
		double sum = 0;
		vcs_.read_lock();
		for (size_t i = 0; i < vcs_.size_locked(); ++i) {
			sum += vcs_.ref_locked(i)->WeightSum();
		}
		vcs_.read_unlock();
		return sum;
	}
	void decayAll() {
		size_t size = vcs_.size();
		for (size_t i = 0; i < size; ++i) {
			vcs_.read_copy(i)->Decay();
		}
	}
	void updateWeightSum() {
		double sum = weightSum();
		if (sum >= weight_sum_max_) {
			std::cerr << "Decay: " << sum << std::endl;
			decayAll();
		}
	}
	void addHotness(size_t tier, const rocksdb::Slice& key, size_t vlen,
			double weight) {
		VisCnts* vc = vcs_.read_copy(tier);
		vc->Access(key, vlen, weight);
		std::tuple<> v;
		auto status = notify_weight_change_.try_pop(v);
		if (boost::fibers::channel_op_status::success == status) {
			updateWeightSum();
		}
	}

	rcu_vector_bp<VisCnts*> vcs_;
	static_assert(!decltype(vcs_)::need_register_thread());
	static_assert(!decltype(vcs_)::need_unregister_thread());
	const rocksdb::Comparator *ucmp_;
	const char *dir_;
	bool create_if_missing_;
	int tier0_last_level_;
	double weight_sum_max_;
	boost::fibers::buffered_channel<std::tuple<>> notify_weight_change_;

	rcu_vector_bp<std::atomic<size_t> *> accessed_;
	static_assert(!decltype(accessed_)::need_register_thread());
	static_assert(!decltype(accessed_)::need_unregister_thread());
	std::atomic<size_t> new_iter_cnt_;
	std::atomic<size_t> hot_taken_;
	rcu_vector_bp<std::atomic<size_t> *> add_hotness_cnts_;
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
			// std::cerr << "There is no background work detected for more than 2 seconds. Exiting...\n";
			break;
		}
	}
}

template <typename T>
void print_vector(const std::vector<T>& v) {
	std::cerr << "{";
	for (size_t i = 0; i < v.size(); ++i) {
		std::cerr << i << ':' << v[i] << ',';
	}
	std::cerr << "}";
}

int main(int argc, char **argv) {
	if (argc != 9) {
		std::cerr << argc << std::endl;
		std::cerr << "Usage:\n";
		std::cerr << "Arg 1: Trace format: plain/ycsb\n";
		std::cerr << "Arg 2: Whether to empty the directories.\n";
		std::cerr << "\t1: Empty the directories first.\n";
		std::cerr << "\t0: Leave the directories as they are.\n";
		std::cerr << "Arg 3: Method to pick SST to compact"
			" (rocksdb::CompactionPri)\n";
		std::cerr << "Arg 4: Delta in bytes\n";
		std::cerr << "Arg 5: Use O_DIRECT for user and compaction reads?\n";
		std::cerr << "\t1: Yes\n";
		std::cerr << "\t0: No\n";
		std::cerr << "Arg 6: Path to database\n";
		std::cerr << "Arg 7: db_paths, for example: "
			"\"{{/tmp/sd,100000000},{/tmp/cd,1000000000}}\"\n";
		std::cerr << "Arg 8: Path to VisCnts\n";
		return -1;
	}
	rocksdb::Options options;

	std::string format(argv[1]);
	bool empty_directories_first = (argv[2][0] == '1');

	char compaction_pri = argv[3][0];
	crash_if(argv[3][1] != 0);
	crash_if(compaction_pri < '0');
	compaction_pri -= '0';

	double delta = atof(argv[4]);
	options.use_direct_reads = (argv[5][0] == '1');
	std::string db_path = std::string(argv[6]);
	std::string db_paths(argv[7]);
	const char *viscnts_path = argv[8];

	options.db_paths = decode_db_paths(db_paths);
	options.compaction_pri =
		static_cast<rocksdb::CompactionPri>(compaction_pri);

	options.write_buffer_size = 1 << 20;
	options.target_file_size_base = 1 << 20;
	options.max_bytes_for_level_base = 4 * options.target_file_size_base;

	if (empty_directories_first) {
		std::cerr << "Emptying directories\n";
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
		new RouterVisCnts(options.comparator, first_cd_level - 1, viscnts_path,
			delta, true);
	options.compaction_router = router;

	rocksdb::DB *db;
	auto s = rocksdb::DB::Open(options, db_path, &db);
	if (!s.ok()) {
		std::cerr << "Creating database\n";
		options.create_if_missing = true;
		s = rocksdb::DB::Open(options, db_path, &db);
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

	auto accessed = router->accessed();
	std::cerr << "Accessed: ";
	print_vector(accessed);
	std::cerr << std::endl;

	std::cerr << "Hot taken: " << router->hot_taken() << std::endl;
	std::cerr << "New iterator count: " << router->new_iter_cnt() << std::endl;

	auto add_hotness_cnts = router->add_hotness_cnts();
	std::cerr << "Add hotness counts: ";
	print_vector(add_hotness_cnts);
	std::cerr << std::endl;

	std::ofstream viscnts_out("viscnts.json");
	viscnts_out << router->sprint_viscnts() << std::endl;

	auto router_timers = router->CollectTimers();
	for (const auto& timer : router_timers) {
		std::cerr << timer.name << ": count " << timer.count <<
			", total " << timer.nsec << "ns,\n";
	}

	auto router_timers_per_level = router->CollectTimersInAllLevels();
	for (size_t level = 0; level < router_timers_per_level.size(); ++level) {
		std::cerr << "{level: " << level << ", timers: [\n";
		for (const auto& timer : router_timers_per_level[level]) {
			std::cerr << timer.name << ": count " << timer.count <<
				", total " << timer.nsec << "ns,\n";
		}
		std::cerr << "]},";
	}
	std::cerr << std::endl;

	timers.Print(std::cerr);

	delete db;
	delete router;

	return ret;
}
