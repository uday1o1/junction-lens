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

struct NllResult {
  double value;
  std::size_t saturation_count;
};

[[nodiscard]] RatioResult SafeRatio(double numerator, double denominator);

[[nodiscard]] bool ReachableWithinHops(
    const std::vector<std::pair<std::uint64_t, std::uint64_t>>& edges, std::uint64_t source,
    std::uint64_t target, std::size_t maximum_hops);

[[nodiscard]] double EndpointGap(const std::array<double, 3>& source_endpoint,
                                 const std::array<double, 3>& target_endpoint);

[[nodiscard]] double LinearQuantile(std::vector<double> values, double probability);

[[nodiscard]] RatioResult StateFlipRate(const std::vector<bool>& states);
[[nodiscard]] double BinaryBrier(const std::vector<double>& probabilities,
                                 const std::vector<std::uint8_t>& outcomes);
[[nodiscard]] NllResult BinaryNll(const std::vector<double>& probabilities,
                                  const std::vector<std::uint8_t>& outcomes);
[[nodiscard]] RatioResult MarginalLaplaceCoverage90(const std::vector<double>& residuals,
                                                    const std::vector<double>& scales,
                                                    const std::vector<double>& factors);

}  // namespace junctionlens::eval
