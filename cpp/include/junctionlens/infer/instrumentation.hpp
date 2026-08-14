#pragma once

#include <cstdint>
#include <string_view>

#if defined(JUNCTIONLENS_ENABLE_NVTX)
#include <nvtx3/nvToolsExt.h>
#endif

namespace junctionlens::infer {

[[nodiscard]] std::uint64_t MonotonicNowNanoseconds() noexcept;
[[nodiscard]] std::string_view MonotonicClockSource() noexcept;
[[nodiscard]] std::uint64_t PeakResidentHostBytes() noexcept;

[[nodiscard]] inline double ElapsedMilliseconds(const std::uint64_t started,
                                                const std::uint64_t finished) noexcept {
  return finished >= started ? static_cast<double>(finished - started) / 1'000'000.0 : 0.0;
}

class NvtxRange final {
 public:
  explicit NvtxRange(const std::string_view name) noexcept {
#if defined(JUNCTIONLENS_ENABLE_NVTX)
    nvtxRangePushA(name.data());
#else
    static_cast<void>(name);
#endif
  }

  NvtxRange(const NvtxRange&) = delete;
  NvtxRange& operator=(const NvtxRange&) = delete;

  ~NvtxRange() {
#if defined(JUNCTIONLENS_ENABLE_NVTX)
    nvtxRangePop();
#endif
  }
};

}  // namespace junctionlens::infer
