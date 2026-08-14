#include "junctionlens/eval/metrics.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>

namespace junctionlens::eval {
namespace {

void RequireFinite(double value, const char* label) {
  if (!std::isfinite(value)) {
    throw std::invalid_argument(std::string(label) + " must be finite");
  }
}

}  // namespace

RatioResult SafeRatio(double numerator, double denominator) {
  RequireFinite(numerator, "numerator");
  RequireFinite(denominator, "denominator");
  if (numerator < 0.0 || denominator < 0.0 || numerator > denominator) {
    throw std::invalid_argument("ratio counts must satisfy 0 <= numerator <= denominator");
  }
  return RatioResult{
      .numerator = numerator,
      .denominator = denominator,
      .value = denominator == 0.0 ? std::nullopt : std::optional<double>{numerator / denominator},
  };
}

bool ReachableWithinHops(const std::vector<std::pair<std::uint64_t, std::uint64_t>>& edges,
                         std::uint64_t source, std::uint64_t target, std::size_t maximum_hops) {
  if (source == 0U || target == 0U || maximum_hops == 0U) {
    return false;
  }
  std::unordered_map<std::uint64_t, std::vector<std::uint64_t>> adjacency;
  for (const auto& [edge_source, edge_target] : edges) {
    if (edge_source == 0U || edge_target == 0U) {
      throw std::invalid_argument("graph IDs must be nonzero");
    }
    adjacency[edge_source].push_back(edge_target);
  }
  std::unordered_set<std::uint64_t> visited{source};
  std::vector<std::uint64_t> frontier{source};
  for (std::size_t hop = 0U; hop < maximum_hops; ++hop) {
    std::vector<std::uint64_t> next;
    for (const auto node : frontier) {
      for (const auto candidate : adjacency[node]) {
        if (candidate == target) {
          return true;
        }
        if (visited.insert(candidate).second) {
          next.push_back(candidate);
        }
      }
    }
    frontier = std::move(next);
    if (frontier.empty()) {
      break;
    }
  }
  return false;
}

double EndpointGap(const std::array<double, 3>& source_endpoint,
                   const std::array<double, 3>& target_endpoint) {
  double squared = 0.0;
  for (std::size_t index = 0U; index < source_endpoint.size(); ++index) {
    RequireFinite(source_endpoint[index], "source endpoint");
    RequireFinite(target_endpoint[index], "target endpoint");
    const double delta = source_endpoint[index] - target_endpoint[index];
    squared += delta * delta;
  }
  return std::sqrt(squared);
}

double LinearQuantile(std::vector<double> values, double probability) {
  RequireFinite(probability, "probability");
  if (values.empty() || probability < 0.0 || probability > 1.0) {
    throw std::invalid_argument("quantile requires values and probability in [0, 1]");
  }
  for (const double value : values) {
    RequireFinite(value, "quantile value");
  }
  std::sort(values.begin(), values.end());
  const double position = static_cast<double>(values.size() - 1U) * probability;
  const auto lower = static_cast<std::size_t>(std::floor(position));
  const auto upper = static_cast<std::size_t>(std::ceil(position));
  const double fraction = position - static_cast<double>(lower);
  return values[lower] * (1.0 - fraction) + values[upper] * fraction;
}

RatioResult StateFlipRate(const std::vector<bool>& states) {
  if (states.size() < 2U) {
    return SafeRatio(0.0, 0.0);
  }
  std::size_t changes = 0U;
  for (std::size_t index = 1U; index < states.size(); ++index) {
    changes += static_cast<std::size_t>(states[index] != states[index - 1U]);
  }
  return SafeRatio(static_cast<double>(changes), static_cast<double>(states.size() - 1U));
}

double BinaryBrier(const std::vector<double>& probabilities,
                   const std::vector<std::uint8_t>& outcomes) {
  if (probabilities.empty() || probabilities.size() != outcomes.size()) {
    throw std::invalid_argument("binary Brier populations must be nonempty and aligned");
  }
  double total = 0.0;
  for (std::size_t index = 0U; index < probabilities.size(); ++index) {
    const double probability = probabilities[index];
    RequireFinite(probability, "probability");
    if (probability < 0.0 || probability > 1.0 || outcomes[index] > 1U) {
      throw std::invalid_argument("binary Brier observation is invalid");
    }
    const double difference = probability - static_cast<double>(outcomes[index]);
    total += difference * difference;
  }
  return total / static_cast<double>(probabilities.size());
}

NllResult BinaryNll(const std::vector<double>& probabilities,
                    const std::vector<std::uint8_t>& outcomes) {
  constexpr double epsilon = 1.0e-7;
  if (probabilities.empty() || probabilities.size() != outcomes.size()) {
    throw std::invalid_argument("binary NLL populations must be nonempty and aligned");
  }
  double total = 0.0;
  std::size_t saturation_count = 0U;
  for (std::size_t index = 0U; index < probabilities.size(); ++index) {
    const double probability = probabilities[index];
    RequireFinite(probability, "probability");
    if (probability < 0.0 || probability > 1.0 || outcomes[index] > 1U) {
      throw std::invalid_argument("binary NLL observation is invalid");
    }
    saturation_count +=
        static_cast<std::size_t>(probability < epsilon || probability > 1.0 - epsilon);
    const double clipped = std::clamp(probability, epsilon, 1.0 - epsilon);
    const double outcome = static_cast<double>(outcomes[index]);
    total -= outcome * std::log(clipped) + (1.0 - outcome) * std::log(1.0 - clipped);
  }
  return NllResult{.value = total / static_cast<double>(probabilities.size()),
                   .saturation_count = saturation_count};
}

RatioResult MarginalLaplaceCoverage90(const std::vector<double>& residuals,
                                      const std::vector<double>& scales,
                                      const std::vector<double>& factors) {
  if (residuals.empty() || residuals.size() != scales.size() ||
      residuals.size() != factors.size()) {
    throw std::invalid_argument("geometry uncertainty populations must be nonempty and aligned");
  }
  std::size_t covered = 0U;
  for (std::size_t index = 0U; index < residuals.size(); ++index) {
    RequireFinite(residuals[index], "residual");
    RequireFinite(scales[index], "scale");
    RequireFinite(factors[index], "factor");
    if (scales[index] <= 0.0 || factors[index] <= 0.0) {
      throw std::invalid_argument("geometry uncertainty scales and factors must be positive");
    }
    const double half_width = scales[index] * factors[index] * std::log(10.0);
    covered += static_cast<std::size_t>(std::abs(residuals[index]) <= half_width);
  }
  return SafeRatio(static_cast<double>(covered), static_cast<double>(residuals.size()));
}

}  // namespace junctionlens::eval
