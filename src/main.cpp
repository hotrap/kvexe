#include <iostream>
#include <filesystem>
#include <fstream>

#include "rocksdb/db.h"

using namespace std;

// void empty_directory(std::string dir_path) {
// 	for (auto& path : std::filesystem::directory_iterator(dir_path)) {
// 		std::filesystem::remove_all(path);
// 	}
// }

bool is_empty_directory(std::string dir_path) {
	auto it = std::filesystem::directory_iterator(dir_path);
	return it == std::filesystem::end(it);
}

int work(rocksdb::DB *db, const std::string &kvops_path, ostream &ans_out) {
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

int main(int argc, char **argv) {
	if (argc != 4) {
		std::cout << "Usage:\n";
		std::cout << "Arg 1: Path to database\n";
		std::cout << "Arg 2: Path to KV operation trace file\n";
		std::cout << "Arg 3: Path to save output\n";
		return -1;
	}
	std::string db_path = std::string(argv[1]);
	std::string kvops_path = std::string(argv[2]);
	std::string ans_out_path = std::string(argv[3]);
	rocksdb::Options options;
	// options.db_paths = {{"/tmp/sd", 10000000}, {"/tmp/cd", 100000000}}; // 10MB, 100MB
	options.db_paths = {{"/tmp/sd", 100000000}, {"/tmp/cd", 1000000000}}; // 100MB, 1GB
	// options.db_paths = {{"/tmp/sd", 10000000000}, {"/tmp/cd", 100000000000}}; // 10GB, 100GB

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
