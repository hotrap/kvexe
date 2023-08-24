#ifndef TIMERS_H_
#define TIMERS_H_

#include <atomic>
#include <cstdint>
#include <memory>
#include <ostream>
#include <rcu_vector_bp.hpp>
#include <rusty/time.h>
#include <vector>

namespace counter_timer {

struct CountTime {
	uint64_t count;
	rusty::time::Duration time;
	CountTime operator+(const CountTime &rhs) const {
		return CountTime{
			.count = count + rhs.count,
			.time = time + rhs.time,
		};
	}
};

class TimerGuard;
class Timer {
public:
	Timer() : count_(0), nsec_(0) {}
	uint64_t count() const { return count_.load(); }
	rusty::time::Duration time() const {
		return rusty::time::Duration::from_nanos(nsec_.load());
	}
	void add(rusty::time::Duration time) const {
		count_.fetch_add(1, std::memory_order_relaxed);
		nsec_.fetch_add(time.as_nanos(), std::memory_order_relaxed);
	}
	TimerGuard start() const;
	CountTime status_nonatomic() const {
		return CountTime{
			.count = count(),
			.time = time(),
		};
	}
private:
	mutable std::atomic<uint64_t> count_;
	mutable std::atomic<uint64_t> nsec_;
};

class TimerGuard {
public:
	~TimerGuard() {
		if (timer_ == nullptr) return;
		timer_->add(rusty::time::Instant::now() - start_time_);
	}
private:
	friend class Timer;
	TimerGuard(
		const Timer &timer
	) : timer_(&timer), start_time_(rusty::time::Instant::now()) {}
	const Timer *timer_;
	rusty::time::Instant start_time_;
};

inline TimerGuard Timer::start() const { return TimerGuard(*this); }

class Timers {
public:
	Timers(size_t num) : timers_(num) {}
	size_t len() const { return timers_.size(); }
	const Timer &timer(size_t type) const { return timers_[type]; }
private:
	std::vector<Timer> timers_;
};

class TimersVector {
public:
	TimersVector(size_t num_types) : num_types_(num_types) {}
	TimersVector(const TimersVector &) = delete;
	TimersVector &operator=(const TimersVector &) = delete;
	TimersVector(TimersVector &&) = delete;
	TimersVector &operator=(TimersVector &&) = delete;
	~TimersVector() {
		size_t size = v_.size_locked();
		for (size_t i = 0; i < size; ++i) {
			delete v_.ref_locked(i);
		}
	}
	size_t len() const { return v_.size(); }
	const Timers &timers(size_t i) const {
		if (v_.size() <= i) {
			v_.lock();
			while (v_.size_locked() <= i) {
				v_.push_back_locked(new Timers(num_types_));
			}
			v_.unlock();
		}
		return *v_.read_copy(i);
	}
	const Timer &timer(size_t i, size_t type) const {
		return timers(i).timer(type);
	}
private:
	mutable rcu_vector_bp<Timers *> v_;
	static_assert(!decltype(v_)::need_register_thread());
	static_assert(!decltype(v_)::need_unregister_thread());
	const size_t num_types_;
};

template <typename Type>
class TypedTimers {
public:
	TypedTimers(size_t num) : timers_(num) {}
	const Timers &timers() const { return timers_; }
	const Timer &timer(Type type) const {
		return timers_.timer(static_cast<size_t>(type));
	}
private:
	Timers timers_;
};

template <typename Type>
class TypedTimersVector {
public:
	TypedTimersVector(size_t num_types) : v_(num_types) {}
	size_t len() const { return v_.len(); }
	const Timers &timers(size_t i) const { return v_.timers(i); }
	const Timer &timer(size_t i, Type type) const {
		return v_.timer(i, static_cast<size_t>(type));
	}
private:
	TimersVector v_;
};
} // namespace counter_timer

#endif // TIMERS_H_
