#ifndef TIMERS_H_
#define TIMERS_H_

#include "rcu_vector_bp.hpp"

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
	};
	Timers(size_t num) : timers_(num) {}
	void Add(size_t type, uint64_t nsec) {
		timers_[type].add(nsec);
	}
	static auto Start() {
	    return std::chrono::steady_clock().now();
	}
	void Stop(size_t type, std::chrono::steady_clock::time_point start_time) {
		auto end_time = std::chrono::steady_clock().now();
		auto nsec = std::chrono::duration_cast<std::chrono::nanoseconds>(
				end_time - start_time).count();
		Add(type, nsec);
	}
	std::vector<Status> TimerCollect() {
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

class TimersPerLevel {
public:
	TimersPerLevel(size_t num_timers_in_each_level)
	:	num_timers_in_each_level_(num_timers_in_each_level) {}
	TimersPerLevel(const TimersPerLevel&) = delete;
	TimersPerLevel& operator=(const TimersPerLevel&) = delete;
	TimersPerLevel(TimersPerLevel&&) = delete;
	TimersPerLevel& operator=(TimersPerLevel&&) = delete;
	~TimersPerLevel() {
		size_t size = v_.size_locked();
		for (size_t i = 0; i < size; ++i)
			delete v_.ref_locked(i);
	}
	void Stop(size_t level, size_t type,
		std::chrono::steady_clock::time_point start_time
	) {
		if (v_.size() <= level) {
			v_.lock();
			while (v_.size_locked() <= level)
				v_.push_back_locked(new Timers(num_timers_in_each_level_));
			v_.unlock();
		}
		v_.read_copy(level)->Stop(type, start_time);
	}
	auto Collect() -> std::vector<std::vector<Timers::Status>> {
		std::vector<std::vector<Timers::Status>> ret;
		size_t num_level = v_.size();
		for (size_t i = 0; i < num_level; ++i)
			ret.push_back(v_.read_copy(i)->TimerCollect());
		return ret;
	}
private:
	rcu_vector_bp<Timers *> v_;
	static_assert(!decltype(v_)::need_register_thread());
	static_assert(!decltype(v_)::need_unregister_thread());
	size_t num_timers_in_each_level_;
};

template <typename Type, const char **names>
class TypedTimers {
public:
	TypedTimers() : timers_(NUM) {}
	void Stop(Type type, std::chrono::steady_clock::time_point start_time) {
		timers_.Stop(static_cast<size_t>(type), start_time);
	}
	void Print(std::ostream& out) {
		auto timers = timers_.TimerCollect();
		for (size_t i = 0; i < NUM; ++i) {
			out << names[i] << ": count " << timers[i].count << ","
				" total " << timers[i].nsec << "ns\n";
		}
	}
private:
	static constexpr size_t NUM = static_cast<size_t>(Type::kEnd);
	Timers timers_;
};

template <typename Type>
class TypedTimersPerLevel {
public:
	TypedTimersPerLevel() : v_(static_cast<size_t>(Type::kEnd)) {}
	void Stop(size_t level, Type type,
		std::chrono::steady_clock::time_point start_time
	) {
		v_.Stop(level, static_cast<size_t>(type), start_time);
	}
	auto Collect() -> std::vector<std::vector<Timers::Status>> {
		return v_.Collect();
	}
private:
	TimersPerLevel v_;
};

#endif // TIMERS_H_
