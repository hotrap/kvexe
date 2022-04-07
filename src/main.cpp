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
	rocksdb::CompactionRouter::Decision
	Route(int, const rocksdb::Slice &) override {
		if (dis_(gen_) >= prob_to_next_)
			return rocksdb::CompactionRouter::Decision::kNextLevel;
		else
			return rocksdb::CompactionRouter::Decision::kCurrentLevel;
	}
	const char *Name() const override {
		return "RouterTrivial";
	}
private:
	float prob_to_next_;
	std::mt19937 gen_;
	std::uniform_real_distribution<float> dis_;
};

int main(int argc, char **argv) {
	if (argc != 5) {
		std::cout << argc << std::endl;
		std::cout << "Usage:\n";
		std::cout << "Arg 1: Path to database\n";
		std::cout << "Arg 2: db_paths, for example: "
			"\"{{/tmp/sd,100000000},{/tmp/cd,1000000000}}\"\n";
		std::cout << "Arg 3: Path to KV operation trace file\n";
		std::cout << "Arg 4: Path to save output\n";
		return -1;
	}
	std::string db_path = std::string(argv[1]);
	std::string db_paths(argv[2]);
	std::string kvops_path = std::string(argv[3]);
	std::string ans_out_path = std::string(argv[4]);
	rocksdb::Options options;

	options.db_paths = decode_db_paths(db_paths);
	// options.compaction_router = new RouterTrivial;
	options.compaction_router = new RouterProb(0.5, 233);

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
