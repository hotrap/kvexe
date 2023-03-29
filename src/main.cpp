#include "rocksdb/compaction_router.h"
#include "timers.h"

#include <atomic>
#include <cstdlib>
#include <iostream>
#include <filesystem>
#include <fstream>
#include <random>
#include <set>
#include <thread>
#include <queue>

#include "rocksdb/db.h"
#include "rocksdb/statistics.h"
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

enum class TimerType : size_t {
	kInsert = 0,
	kRead,
	kUpdate,
	kPut,
	kGet,
	kInput,
	kOutput,
	kSerialize,
	kDeserialize,
	kRangeHotSize,
	kDecay,
	kNextHot,
	kCountAccessHotPerTier,
	kEnd,
};
const char *timer_names[] = {
	"Insert",
	"Read",
	"Update",
	"Put",
	"Get",
	"Input",
	"Output",
	"Serialize",
	"Deserialize",
	"RangeHotSize",
	"Decay",
	"NextHot",
	"CountAccessHotPerTier",
};
TypedTimers<TimerType, timer_names> timers;

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
		in >> op;
		if (!in) {
			break;
		}
		if (op == "INSERT") {
			auto insert_start = Timers::Start();
			auto input_start = Timers::Start();
			handle_table_name(in);
			std::string key;
			in >> key;
			rocksdb::Slice key_slice(key);
			auto field_values = read_field_values(in);
			timers.Stop(TimerType::kInput, input_start);
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
			auto read_start = Timers::Start();
			auto input_start = Timers::Start();
			handle_table_name(in);
			std::string key;
			in >> key;
			rocksdb::Slice key_slice(key);
			auto fields = read_fields(in);
			timers.Stop(TimerType::kInput, input_start);
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
			timers.Stop(TimerType::kRead, read_start);
		} else if (op == "UPDATE") {
			auto update_start = Timers::Start();
			auto input_start = Timers::Start();
			handle_table_name(in);
			std::string key;
			in >> key;
			rocksdb::Slice key_slice(key);
			auto updates = read_field_values(in);
			timers.Stop(TimerType::kInput, input_start);
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

enum class PerLevelTimerType : size_t {
	kAccess = 0,
	kEnd,
};
const char *per_level_timer_names[] = {
	"Access",
};

enum class PerTierTimerType : size_t {
	kAddHostness = 0,
	kEnd,
};
const char *per_tier_timer_names[] = {
	"AddHotness",
};
static constexpr uint64_t MASK_COUNT_ACCESS_HOT_PER_TIER = 0x1;

class RouterVisCnts : public rocksdb::CompactionRouter {
public:
	RouterVisCnts(const rocksdb::Comparator *ucmp, int target_level,
			const char *dir, double weight_sum_max, bool create_if_missing,
			uint64_t switches)
		:	switches_(switches),
			ucmp_(ucmp),
			dir_(dir),
			create_if_missing_(create_if_missing),
			tier0_last_level_(target_level),
			weight_sum_max_(weight_sum_max),
			notify_weight_change_(2),
			new_iter_cnt_(0),
			count_access_hot_per_tier_{0, 0} {}
	~RouterVisCnts() {
		size_t size = vcs_.size_locked();
		for (size_t i = 0; i < size; ++i) {
			delete vcs_.ref_locked(i);
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
		auto start = Timers::Start();
		addHotness(tier, key, vlen, weight);
		per_tier_timers_.Stop(tier, PerTierTimerType::kAddHostness, start);
	}
	void Access(int level, const rocksdb::Slice& key, size_t vlen)
			override {
		auto start = Timers::Start();
		if (level < tier0_last_level_) {
			per_level_timers_.Stop(level, PerLevelTimerType::kAccess, start);
			return;
		}
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
		per_level_timers_.Stop(level, PerLevelTimerType::kAccess, start);
		if (switches_ & MASK_COUNT_ACCESS_HOT_PER_TIER) {
			auto start_time = Timers::Start();
			size_t num_tiers = vcs_.size();
			assert(num_tiers <= 2);
			for (size_t i = 0; i < num_tiers; ++i) {
				VisCnts *vc = vcs_.read_copy(i);
				if (vc->IsHot(key))
					count_access_hot_per_tier_[i].fetch_add(1);
			}
			timers.Stop(TimerType::kCountAccessHotPerTier, start_time);
		}
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
		auto start_time = Timers::Start();
		auto it = (VisCnts::Iter*)iter;
		const rocksdb::HotRecInfo *ret = it->Next();
		timers.Stop(TimerType::kNextHot, start_time);
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
		auto start = Timers::Start();
		size_t ret = vcs_.read_copy(tier)->RangeHotSize(smallest, largest);
		timers.Stop(TimerType::kRangeHotSize, start);
		return ret;
	}
	std::vector<std::vector<Timers::Status>> per_level_timers() {
		return per_level_timers_.Collect();
	}
	std::vector<std::vector<Timers::Status>> per_tier_timers() {
		return per_tier_timers_.Collect();
	}
	size_t new_iter_cnt() {
		return new_iter_cnt_.load(std::memory_order_relaxed);
	}
	std::vector<size_t> hit_count() {
		std::vector<size_t> ret;
		for (size_t i = 0; i < 2; ++i)
			ret.push_back(count_access_hot_per_tier_[i].load(
				std::memory_order_relaxed));
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
			auto start = Timers::Start();
			decayAll();
			timers.Stop(TimerType::kDecay, start);
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

	const uint64_t switches_;
	rcu_vector_bp<VisCnts*> vcs_;
	static_assert(!decltype(vcs_)::need_register_thread());
	static_assert(!decltype(vcs_)::need_unregister_thread());
	const rocksdb::Comparator *ucmp_;
	const char *dir_;
	bool create_if_missing_;
	int tier0_last_level_;
	double weight_sum_max_;
	boost::fibers::buffered_channel<std::tuple<>> notify_weight_change_;

	std::atomic<size_t> new_iter_cnt_;
	std::atomic<size_t> count_access_hot_per_tier_[2];
	TypedTimersPerLevel<PerLevelTimerType>
		per_level_timers_;
	TypedTimersPerLevel<PerTierTimerType>
		per_tier_timers_;
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

auto AggregateTimers(
	const std::vector<std::vector<rocksdb::TimerStatus>>& timers_per_level
) -> std::vector<rocksdb::TimerStatus> {
	size_t num_levels = timers_per_level.size();
	if (num_levels == 0)
		return std::vector<rocksdb::TimerStatus>();
	size_t num_timers = timers_per_level[0].size();
	std::vector<rocksdb::TimerStatus> ret = timers_per_level[0];
	for (size_t level = 1; level < num_levels; ++level) {
		const auto& timers = timers_per_level[level];
		assert(timers.size() == num_timers);
		for (size_t i = 0; i < num_timers; ++i) {
			assert(strcmp(ret[i].name, timers[i].name) == 0);
			ret[i].count += timers[i].count;
			ret[i].nsec += timers[i].nsec;
		}
	}
	return ret;
}

auto AggregateTimers(
	const std::vector<std::vector<Timers::Status>>& timers_per_level
) -> std::vector<Timers::Status> {
	size_t num_levels = timers_per_level.size();
	if (num_levels == 0)
		return std::vector<Timers::Status>();
	size_t num_timers = timers_per_level[0].size();
	std::vector<Timers::Status> ret = timers_per_level[0];
	for (size_t level = 1; level < num_levels; ++level) {
		const auto& timers = timers_per_level[level];
		for (size_t i = 0; i < num_timers; ++i) {
			ret[i].count += timers[i].count;
			ret[i].nsec += timers[i].nsec;
		}
	}
	return ret;
}

int main(int argc, char **argv) {
	if (argc < 9 || argc > 10) {
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
		std::cerr << "Arg 9: Switch mask (default is 0)\n";
		std::cerr << "\t0x1: count access hot per tier\n";
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
	uint64_t switches;
	if (argc < 10)
		switches = 0;
	else
		switches = strtoul(argv[9], NULL, 0);

	options.db_paths = decode_db_paths(db_paths);
	options.compaction_pri =
		static_cast<rocksdb::CompactionPri>(compaction_pri);
	options.statistics = rocksdb::CreateDBStatistics();

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
			delta, true, switches);
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
	std::cerr << rocksdb_stats << std::endl;

	std::cerr << "New iterator count: " << router->new_iter_cnt() << std::endl;
	if (switches & MASK_COUNT_ACCESS_HOT_PER_TIER) {
		auto counters = router->hit_count();
		assert(counters.size() == 2);
		std::cerr << "Access hot per tier: " << counters[0] << ' ' <<
			counters[1] << std::endl;
	}

	std::ofstream viscnts_out("viscnts.json");
	viscnts_out << router->sprint_viscnts() << std::endl;

	auto router_timers = router->CollectTimers();
	for (const auto& timer : router_timers) {
		std::cerr << timer.name << ": count " << timer.count <<
			", total " << timer.nsec << "ns,\n";
	}

	auto per_tier_timers = router->per_tier_timers();
	for (size_t tier = 0; tier < per_tier_timers.size(); ++tier) {
		std::cerr << "{tier: " << tier << ", timers: [\n";
		const auto& timers = per_tier_timers[tier];
		for (size_t type = 0; type < timers.size(); ++type) {
			std::cerr << per_tier_timer_names[tier] << ": "
				"count " << timers[type].count << ", "
				"total " << timers[type].nsec << "ns,\n";
		}
		std::cerr << "]},";
	}
	std::cerr << std::endl;

	auto per_level_timers = router->per_level_timers();
	for (size_t level = 0; level < per_level_timers.size(); ++level) {
		std::cerr << "{level: " << level << ", timers: [\n";
		const auto& timers = per_level_timers[level];
		for (size_t type = 0; type < timers.size(); ++type) {
			std::cerr << per_level_timer_names[type] << ": "
				"count " <<  timers[type].count << ", "
				"total " << timers[type].nsec << "ns,\n";
		}
		std::cerr << "]},";
	}
	std::cerr << std::endl;

	std::cerr << "In all levels: [\n";
	std::vector<Timers::Status> router_timers_in_all_levels =
		AggregateTimers(per_level_timers);
	for (size_t i = 0; i < router_timers_in_all_levels.size(); ++i) {
		std::cerr << per_level_timer_names[i] << ": "
			"count " << router_timers_in_all_levels[i].count << ", "
			"total " << router_timers_in_all_levels[i].nsec << "ns,\n";
	}
	std::cerr << "]\n";

	timers.Print(std::cerr);

	delete db;
	delete router;

	return ret;
}
