#ifndef TIMERS_H_
#define TIMERS_H_

#include <atomic>
#include <chrono>
#include <cstdint>
#include <memory>
#include <ostream>
#include <vector>

class Timers {
public:
	struct Status {
		uint64_t count;
		uint64_t nsec;
		Status operator+(const Status& rhs) const {
			return Status{
				.count = count + rhs.count,
				.nsec = nsec + rhs.nsec,
			};
		}
	};
	Timers(size_t num) : timers_(num) {}
	void Add(size_t type, uint64_t nsec) {
		timers_[type].add(nsec);
	}
	static auto Start() {
	    return std::chrono::steady_clock::now();
	}
	void Stop(size_t type, std::chrono::steady_clock::time_point start_time) {
		auto end_time = std::chrono::steady_clock::now();
		auto nsec = std::chrono::duration_cast<std::chrono::nanoseconds>(
				end_time - start_time).count();
		Add(type, nsec);
	}
	std::vector<Status> Collect() {
		std::vector<Status> timers;
		for (const Timer& timer : timers_)
			timers.push_back(Status{timer.count(), timer.nsec()});
		return timers;
	}
private:
	class Timer {
	public:
		Timer() : count_(0), nsec_(0) {}
		uint64_t count() const { return count_.load(); }
		uint64_t nsec() const { return nsec_.load(); }
		void add(uint64_t nsec) {
			count_.fetch_add(1, std::memory_order_relaxed);
			nsec_.fetch_add(nsec, std::memory_order_relaxed);
		}
	private:
		std::atomic<uint64_t> count_;
		std::atomic<uint64_t> nsec_;
	};
	std::vector<Timer> timers_;
};

template <typename Type>
class TypedTimers {
public:
	TypedTimers() : timers_(NUM) {}
	void Stop(Type type, std::chrono::steady_clock::time_point start_time) {
		timers_.Stop(static_cast<size_t>(type), start_time);
	}
	std::vector<Timers::Status> Collect() {
		return timers_.Collect();
	}
private:
	static constexpr size_t NUM = static_cast<size_t>(Type::kEnd);
	Timers timers_;
};

#endif // TIMERS_H_
