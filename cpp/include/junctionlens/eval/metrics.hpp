#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <utility>
#include <vector>

namespace junctionlens::eval {

struct RatioResult {
  double numerator;
  double denominator;
  std::optional<double> value;
};

[[nodiscard]] RatioResult SafeRatio(double numerator, double denominator);

[[nodiscard]] bool ReachableWithinHops(
    const std::vector<std::pair<std::uint64_t, std::uint64_t>>& edges, std::uint64_t source,
    std::uint64_t target, std::size_t maximum_hops);

[[nodiscard]] double EndpointGap(const std::array<double, 3>& source_endpoint,
                                 const std::array<double, 3>& target_endpoint);

[[nodiscard]] double LinearQuantile(std::vector<double> values, double probability);

[[nodiscard]] RatioResult StateFlipRate(const std::vector<bool>& states);

}  // namespace junctionlens::eval
