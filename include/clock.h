#pragma once

#include <cassert>
#include <chrono>
#include <ratio>

namespace nt {

using std::chrono::duration;
using std::chrono::high_resolution_clock;
using std::chrono::nanoseconds;

class Clock {
  public:
    void start() {
        assert(running == false);
        running = true;
        lastTimePoint = high_resolution_clock::now();
    }

    void stop() {
        const auto now = high_resolution_clock::now();
        assert(running == true);
        running = false;

        duration += now - lastTimePoint;
    }

    void reset() {
        duration = nanoseconds(0);
        running = false;
    }

    template <class ToDuration> double getDuration() {
        return duration_cast<ToDuration>(duration).count();
    }

  private:
    bool running = false;
    duration<double, std::nano> duration = nanoseconds(0);
    high_resolution_clock::time_point lastTimePoint;
};

} // namespace nt
