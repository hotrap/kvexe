#include <iostream>
#include <filesystem>
#include <fstream>
#include <random>

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
			std::cout << level << ' ' << options.db_paths[p].path << std::endl;
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
	std::cout << level << "+ " << options.db_paths[p].path << std::endl;
	return level;
}

// void empty_directory(std::string dir_path) {
// 	for (auto& path : std::filesystem::directory_iterator(dir_path)) {
// 		std::filesystem::remove_all(path);
// 	}
// }

bool is_empty_directory(std::string dir_path) {
	auto it = std::filesystem::directory_iterator(dir_path);
	return it == std::filesystem::end(it);
}

int work(rocksdb::DB *db, const std::string &kvops_path, std::ostream &ans_out) {
	std::ifstream in(kvops_path);
	if (!in) {
		std::cout << "Fail to open " << kvops_path << std::endl;
		return -1;
	}

	std::vector<std::string> ans;
	std::string op;
	while (std::getline(in, op, '|')) {
		if (op == "UPDATE") {
			std::string key, value;
			std::getline(in, key, '|');
			std::getline(in, value);
			rocksdb::Slice key_slice(std::move(key)), value_slice(std::move(value));
			auto s = db->Put(rocksdb::WriteOptions(), key_slice, value_slice);
			if (!s.ok()) {
				std::cout << "Put failed with error: " << s.ToString() << std::endl;
				return -1;
			}
		} else if (op == "GET") {
			std::string key, value;
			std::getline(in, key);
			rocksdb::Slice key_slice(std::move(key));
			auto s = db->Get(rocksdb::ReadOptions(), key_slice, &value);
			if (!s.ok()) {
				std::cout << "Get failed with error: " << s.ToString() << std::endl;
				return -1;
			}
			ans.push_back(std::move(value));
		} else {
			std::cout << "Unrecognized operator " << op << std::endl;
			return -1;
		}
	}
	for (auto &s : ans) {
		ans_out << s << std::endl;
	}
	return 0;
}

class RouterTrivial : public rocksdb::CompactionRouter {
	void Access(const rocksdb::Slice&) override {}
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
	void Access(const rocksdb::Slice&) override {}
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

extern void *VisCntsOpen(const char *path, float delta, bool createIfMissing);
extern int VisCntsAccess(void *ac, const char *key);
extern bool VisCntsIsHot(void *ac, const char *key);

class RouterVisCnts : public rocksdb::CompactionRouter {
public:
	RouterVisCnts(int target_level, const char *path, float delta,
			bool create_if_missing)
	  : ac_(VisCntsOpen(path, delta, create_if_missing)),
	  	target_level_(target_level) {}
	void Access(const rocksdb::Slice& key) override {
		VisCntsAccess(ac_, key.data());
	}
	rocksdb::CompactionRouter::Decision
	Route(int level, const rocksdb::Slice& key) override {
		if (level != target_level_)
			return rocksdb::CompactionRouter::Decision::kNextLevel;
		if (VisCntsIsHot(ac_, key.data())) {
			return rocksdb::CompactionRouter::Decision::kCurrentLevel;
		} else {
			return rocksdb::CompactionRouter::Decision::kNextLevel;
		}
	}
	const char *Name() const override {
		return "RouterVisCnts";
	}
private:
	void *ac_;
	int target_level_;
};

int main(int argc, char **argv) {
	if (argc != 6) {
		std::cout << argc << std::endl;
		std::cout << "Usage:\n";
		std::cout << "Arg 1: Path to database\n";
		std::cout << "Arg 2: db_paths, for example: "
			"\"{{/tmp/sd,100000000},{/tmp/cd,1000000000}}\"\n";
		std::cout << "Arg 3: Path to KV operation trace file\n";
		std::cout << "Arg 4: Path to save output\n";
		std::cout << "Arg 5: Path to VisCnts\n";
		return -1;
	}
	std::string db_path = std::string(argv[1]);
	std::string db_paths(argv[2]);
	std::string kvops_path = std::string(argv[3]);
	std::string ans_out_path = std::string(argv[4]);
	const char *viscnts_path = argv[5];
	rocksdb::Options options;

	options.db_paths = decode_db_paths(db_paths);

	int first_cd_level = predict_level_assignment(options);

	// options.compaction_router = new RouterTrivial;
	// options.compaction_router = new RouterProb(0.5, 233);
	options.compaction_router =
		new RouterVisCnts(first_cd_level - 1, viscnts_path, 1e7, true);

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

	int ret = work(db, kvops_path, ans_out);

	delete db;

	return ret;
}
