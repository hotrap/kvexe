#include <leveldb/db.h>
#include <leveldb/filter_policy.h>
#include <leveldb/table.h>
#include <leveldb/cache.h>
#include <leveldb/env.h>
#include <rusty/keyword.h>
#include <rusty/macro.h>
#include <rusty/primitive.h>
#include <rusty/time.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <boost/fiber/buffered_channel.hpp>
#include <boost/program_options/errors.hpp>
#include <boost/program_options/options_description.hpp>
#include <boost/program_options/parsers.hpp>
#include <boost/program_options/variables_map.hpp>
#include <cctype>
#include <chrono>
#include <counter_timer.hpp>
#include <cstddef>
#include <cstdlib>
#include <deque>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <set>
#include <string>
#include <thread>
#include <queue>
#include <vector>


enum class FormatType {
  Plain,
  YCSB,
};

enum class OpType {
  INSERT,
  READ,
  UPDATE,
};


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

constexpr size_t TIMER_NUM = static_cast<size_t>(TimerType::kEnd);
static const char *timer_names[] = {
    "Insert",      "Read",           "Update",      "Put",
    "Get",         "InputOperation", "InputInsert", "InputRead",
    "InputUpdate", "Output",         "Serialize",   "Deserialize",
};
static_assert(sizeof(timer_names) == TIMER_NUM * sizeof(const char *));
static counter_timer::TypedTimers<TimerType> timers(TIMER_NUM);

constexpr uint64_t MASK_LATENCY = 0x1;
constexpr uint64_t MASK_OUTPUT_ANS = 0x2;

struct Operation {
  OpType type;
  std::string key;
  std::vector<char> value;

  Operation() {}

  Operation(OpType _type, const std::string& _key, const std::vector<char>& _value) :
    type(_type), key(_key), value(_value) {}

  Operation(OpType _type, std::string&& _key, std::vector<char>&& _value) :
    type(_type), key(std::move(_key)), value(std::move(_value)) {}
};

template <typename T>
class BlockChannel {
  std::queue<std::vector<T>> q_;
  std::mutex m_;
  std::condition_variable cv_r_;
  std::condition_variable cv_w_;
  size_t reader_waiting_{0};
  size_t limit_size_{0};
  bool finish_{false};
  bool writer_waiting_{0};
 
 public:
  BlockChannel(size_t limit_size = 4192) : limit_size_(limit_size) {}
  std::vector<T> GetBlock() {
    std::unique_lock lck(m_);
    if (writer_waiting_) {
      if (q_.size() < limit_size_ / 2) {
        cv_w_.notify_one();
      }
    }
    if (q_.size()) {
      auto ret = std::move(q_.front());
      q_.pop();
      return ret;
    }
    if (finish_) {
      return {};
    }
    reader_waiting_ += 1;
    cv_r_.wait(lck, [&]() { return finish_ || !q_.empty(); });
    if (q_.empty() && finish_) {
      return {};
    }
    reader_waiting_ -= 1;
    auto ret = std::move(q_.front());
    q_.pop();
    return ret;
  }

  void PutBlock(std::vector<T>&& block) {
    std::unique_lock lck(m_);
    q_.push(std::move(block));
    if (reader_waiting_) {
      cv_r_.notify_one();
    }
  }

  void PutBlock(const std::vector<T>& block) {
    std::unique_lock lck(m_);
    q_.push(block);
    if (reader_waiting_) {
      cv_r_.notify_one();
    }
    if (q_.size() >= limit_size_) {
      writer_waiting_ = true;
      cv_w_.wait(lck, [&]() { return q_.size() < limit_size_ / 2; });
      writer_waiting_ = false;
    }
  }

  void Finish() {
    finish_ = true;
    cv_r_.notify_all();
  }

};

template<typename T>
class BlockChannelClient {
  BlockChannel<T>* channel_;
  std::vector<T> opblock_;
  size_t opnum_{0};

 public:
  BlockChannelClient(BlockChannel<T>* channel, size_t block_size) : channel_(channel), opblock_(block_size) {}

  void Push(T&& data) {
    opblock_[opnum_++] = std::move(data);
    if (opnum_ == opblock_.size()) {
      channel_->PutBlock(opblock_);
      opnum_ = 0;
    }
  }

  void Flush() {
    channel_->PutBlock(std::vector<T>(opblock_.begin(), opblock_.begin() + opnum_));
    opnum_ = 0;
  }

  void Finish() {
    channel_->Finish();
  }
};

struct WorkOptions {
  FormatType format_type;
  leveldb::DB *db;
  uint64_t switches;
  std::filesystem::path db_path;
  std::atomic<size_t> *progress;
  bool enable_fast_process{false};
  size_t num_threads{1};
  size_t opblock_size{1024};
  size_t num_keys;
};

struct WorkerEnv {
  leveldb::DB *db;
  leveldb::ReadOptions read_options;
  leveldb::WriteOptions write_options;
  bool ignore_notfound{false};
};

void print_ans(std::ofstream &out, std::string value) {
  out << value << '\n';
}

void print_latency(std::ofstream &out, OpType op, uint64_t nanos) {
  switch (op) {
    case OpType::INSERT:
      out << "INSERT";
      break;
    case OpType::READ:
      out << "READ";
      break;
    case OpType::UPDATE:
      out << "UPDATE";
      break;
  }
  out << ' ' << nanos << '\n';
}

class Tester {
  WorkOptions options_;
  WorkerEnv env_;
  BlockChannel<Operation> channel_;
  std::vector<BlockChannel<Operation>> channel_for_workers_;


  size_t parse_counts_{0};
  std::atomic<size_t> notfound_counts_{0};
  
 public:
  Tester(const WorkOptions& option) : options_(option), channel_for_workers_(option.num_threads) {
    env_.db = options_.db;
    env_.read_options = leveldb::ReadOptions();
    env_.write_options = leveldb::WriteOptions();
    if (options_.enable_fast_process) {
      env_.ignore_notfound = true;
    }
  }

  void Test() {
    std::vector<std::thread> threads;

    for (size_t i = 0; i < options_.num_threads; i++) {
      threads.emplace_back([this, i]() { work(i, options_.enable_fast_process ? channel_ : channel_for_workers_[i]); });
    }

    parse();

    for (auto& t : threads) t.join();
  }

  size_t GetOpParseCounts() const {
    return parse_counts_;
  }

  size_t GetNotFoundCounts() const {
    return notfound_counts_.load(std::memory_order_relaxed);
  }

 private:
  void work(size_t id, BlockChannel<Operation>& chan) {
    std::optional<std::ofstream> ans_out =
        options_.switches & MASK_OUTPUT_ANS
            ? std::optional<std::ofstream>(options_.db_path /
                                          ("ans_" + std::to_string(id)))
            : std::nullopt;

    std::optional<std::ofstream> latency_out =
        options_.switches & MASK_LATENCY
            ? std::optional<std::ofstream>(options_.db_path /
                                          ("latency_" + std::to_string(id)))
            : std::nullopt;  
    std::string read_value(4096, 0);
    while (true) {
      auto block = chan.GetBlock();
      if (block.empty()) {
        break;
      }
      size_t local_notfound_counts = 0;
      for (Operation& op : block) {
        if (op.type == OpType::INSERT) {
          do_insert(latency_out, op);
        } else if (op.type == OpType::READ) {
          auto is_notfound = do_read(latency_out, read_value, op);
          // if (ans_out) {
          //   print_ans(ans_out.value(), value);
          // }
          if (is_notfound) {
            local_notfound_counts++;
          }
        } else if (op.type == OpType::UPDATE) {
          do_update(latency_out, op);
        }
        options_.progress->fetch_add(1, std::memory_order_relaxed);
      }
      notfound_counts_ += local_notfound_counts;
    }
  }

  
  void do_insert(std::optional<std::ofstream>& latency, const Operation& insert) {
    auto guard = timers.timer(TimerType::kInsert).start();
    auto put_start = rusty::time::Instant::now();
    auto s = env_.db->Put(env_.write_options, insert.key, leveldb::Slice(insert.value.data(), std::min<size_t>(990, insert.value.size())));
    auto put_time = put_start.elapsed();
    if (!s.ok()) {
      std::string err = s.ToString();
      rusty_panic("INSERT failed with error: %s\n", err.c_str());
    }
    timers.timer(TimerType::kPut).add(put_time);
    if (latency) {
      print_latency(latency.value(), OpType::INSERT, put_time.as_nanos());
    }
  }

  bool do_read(std::optional<std::ofstream>& latency, std::string& read_value, const Operation& read) {
    auto guard = timers.timer(TimerType::kRead).start();
    auto get_start = rusty::time::Instant::now();
    auto s = env_.db->Get(env_.read_options, read.key, &read_value);
    auto get_time = get_start.elapsed();
    bool is_notfound = false;
    if (!s.ok()) {
      if (s.IsNotFound() && env_.ignore_notfound) {
        is_notfound = true;
      } else {
        std::string err = s.ToString();
        rusty_panic("GET failed with error: %s\n", err.c_str());
      }
    }
    timers.timer(TimerType::kGet).add(get_time);
    if (latency) {
      print_latency(latency.value(), OpType::READ, get_time.as_nanos());
    }
    return is_notfound;
  }

  void do_update(std::optional<std::ofstream>& latency, const Operation& update) {
    auto guard = timers.timer(TimerType::kUpdate).start();
    auto put_start = rusty::time::Instant::now();
    auto s = env_.db->Put(env_.write_options, update.key, leveldb::Slice(update.value.data(), std::min<size_t>(990, update.value.size())));
    auto put_time = put_start.elapsed();
    if (!s.ok()) {
      std::string err = s.ToString();
      rusty_panic("Update failed with error: %s\n", err.c_str());
    }
    timers.timer(TimerType::kUpdate).add(put_time);
    if (latency) {
      print_latency(latency.value(), OpType::UPDATE, put_time.as_nanos());
    }
  }

  std::string gen_prism_key(const std::string& key) {
    std::hash<std::string> hasher;
    char a[8] = {0};
    size_t hv = hasher(key) % options_.num_keys;
    for (int i = 7; i >= 0; --i)
      a[i] = hv >> (7 - i) * 8 & 255;
    return std::string(a, 8);
  }

  void parse() {
    std::vector<BlockChannelClient<Operation>> opblocks;
    if (options_.enable_fast_process) {
      opblocks.emplace_back(&channel_, options_.opblock_size);
    } else {
      for (size_t i = 0; i < options_.num_threads; i++) {
        opblocks.emplace_back(&channel_for_workers_[i], options_.opblock_size);
      }
    }

    std::hash<std::string> hasher{};

    while (1) {
      std::string op;
      std::cin >> op;
      if (!std::cin) {
        break;
      }
      if (op == "INSERT") {
        if (options_.format_type == FormatType::YCSB) { 
          handle_table_name(std::cin);
        }
        std::string key;
        std::cin >> key;
        int i = options_.enable_fast_process ? 0 : hasher(key) % options_.num_threads;
        if (options_.format_type == FormatType::Plain) {
          rusty_assert(std::cin.get() == ' ');
          char c;
          std::vector<char> value;
          while ((c = std::cin.get()) != '\n' && c != EOF) {
            value.push_back(c);
          }
          opblocks[i].Push(Operation(OpType::INSERT, gen_prism_key(key), std::move(value)));
        } else {
          opblocks[i].Push(Operation(OpType::INSERT, gen_prism_key(key), read_value(std::cin)));
        }
        parse_counts_++;
      } else if (op == "READ") {
        std::string key;
        if (options_.format_type == FormatType::YCSB) {        
          handle_table_name(std::cin);
          std::cin >> key;
          read_fields_read(std::cin);
        } else {
          std::cin >> key;
        }
        int i = options_.enable_fast_process ? 0 : hasher(key) % options_.num_threads;
        opblocks[i].Push(Operation(OpType::READ, gen_prism_key(key), {}));
        parse_counts_++;

      } else if (op == "UPDATE") {
        if (options_.format_type == FormatType::Plain) {
          rusty_panic("UPDATE in plain format is not supported yet\n");
        }
        handle_table_name(std::cin);
        std::string key;
        std::cin >> key;
        int i = options_.enable_fast_process ? 0 : hasher(key) % options_.num_threads;
        opblocks[i].Push(Operation(OpType::UPDATE, gen_prism_key(key), read_value(std::cin)));
        parse_counts_++;
        
      } else {
        std::cerr << "Ignore line: " << op;
        std::getline(std::cin, op);  // Skip the rest of the line
        std::cerr << op << std::endl;
      }
    }

    for(auto& o : opblocks) {
      o.Flush();
      o.Finish();
    }
  }


  void handle_table_name(std::istream &in) {
    std::string table;
    in >> table;
    rusty_assert(table == "usertable", "Column families not supported yet.");
  }

  std::vector<std::pair<int, std::vector<char>>> read_fields(std::istream &in) {
    std::vector<std::pair<int, std::vector<char>>> ret;
    char c;
    do {
      c = static_cast<char>(in.get());
    } while (isspace(c));
    rusty_assert(c == '[', "Invalid KV trace!");
    rusty_assert(in.get() == ' ', "Invalid KV trace!");
    while (in.peek() != ']') {
      constexpr size_t vallen = 100;
      std::string field;
      std::vector<char> value(vallen);
      while ((c = static_cast<char>(in.get())) != '=') {
        field.push_back(c);
      }
      rusty_assert(field.size() > 5);
      rusty_assert(field.substr(0, 5) == "field");
      int i = std::stoi(field.substr(5));
      rusty_assert(in.read(value.data(), vallen), "Invalid KV trace!");
      rusty_assert(in.get() == ' ', "Invalid KV trace!");
      ret.emplace_back(i, std::move(value));
    }
    in.get();  // ]
    return ret;
  }

  std::vector<char> read_value(std::istream &in) {
    auto fields = read_fields(in);
    std::sort(fields.begin(), fields.end());
    std::vector<char> ret;
    for (int i = 0; i < (int)fields.size(); ++i) {
      rusty_assert(fields[i].first == i);
      ret.insert(ret.end(), fields[i].second.begin(), fields[i].second.end());
    }
    return ret;
  }

  void read_fields_read(std::istream &in) {
    char c;
    do {
      c = static_cast<char>(in.get());
    } while (isspace(c));
    rusty_assert(c == '[', "Invalid KV trace!");
    std::string s;
    std::getline(in, s);
    rusty_assert(s == " <all fields>]",
                "Reading specific fields is not supported yet.");
  }

};
