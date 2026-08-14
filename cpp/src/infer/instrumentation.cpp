#include "junctionlens/infer/instrumentation.hpp"

#include <chrono>
#include <cstdint>

#if defined(__unix__) || defined(__APPLE__)
#include <sys/resource.h>
#include <time.h>
#endif

namespace junctionlens::infer {

std::uint64_t MonotonicNowNanoseconds() noexcept {
#if defined(CLOCK_MONOTONIC_RAW)
  timespec value{};
  if (clock_gettime(CLOCK_MONOTONIC_RAW, &value) == 0) {
    return static_cast<std::uint64_t>(value.tv_sec) * 1'000'000'000U +
           static_cast<std::uint64_t>(value.tv_nsec);
  }
#endif
  return static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                        std::chrono::steady_clock::now().time_since_epoch())
                                        .count());
}

std::string_view MonotonicClockSource() noexcept {
#if defined(CLOCK_MONOTONIC_RAW)
  return "CLOCK_MONOTONIC_RAW";
#else
  return "std::chrono::steady_clock";
#endif
}

std::uint64_t PeakResidentHostBytes() noexcept {
#if defined(__unix__) || defined(__APPLE__)
  rusage usage{};
  if (getrusage(RUSAGE_SELF, &usage) != 0 || usage.ru_maxrss < 0) {
    return 0U;
  }
#if defined(__APPLE__)
  return static_cast<std::uint64_t>(usage.ru_maxrss);
#else
  return static_cast<std::uint64_t>(usage.ru_maxrss) * 1024U;
#endif
#else
  return 0U;
#endif
}

}  // namespace junctionlens::infer
