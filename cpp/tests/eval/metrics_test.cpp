#include "junctionlens/eval/metrics.hpp"

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace {

std::vector<std::string> Split(const std::string& value, char delimiter) {
  std::vector<std::string> result;
  std::istringstream source(value);
  std::string item;
  while (std::getline(source, item, delimiter)) {
    result.push_back(item);
  }
  return result;
}

std::vector<double> Doubles(const std::string& value) {
  std::vector<double> result;
  for (const auto& item : Split(value, ',')) {
    result.push_back(std::stod(item));
  }
  return result;
}

TEST(EvalMetrics, LanguageNeutralGoldens) {
  std::ifstream source(JUNCTIONLENS_CUSTOM_METRIC_GOLDENS);
  ASSERT_TRUE(source.good());
  std::string line;
  std::size_t cases = 0U;
  while (std::getline(source, line)) {
    const auto fields = Split(line, '|');
    ASSERT_FALSE(fields.empty());
    if (fields[0] == "ratio" || fields[0] == "ratio_empty") {
      ASSERT_EQ(fields.size(), 4U);
      const auto result = junctionlens::eval::SafeRatio(std::stod(fields[1]), std::stod(fields[2]));
      if (fields[3] == "null") {
        EXPECT_FALSE(result.value.has_value());
      } else {
        ASSERT_TRUE(result.value.has_value());
        EXPECT_DOUBLE_EQ(*result.value, std::stod(fields[3]));
      }
    } else if (fields[0] == "linear_quantile") {
      ASSERT_EQ(fields.size(), 4U);
      EXPECT_DOUBLE_EQ(junctionlens::eval::LinearQuantile(Doubles(fields[2]), std::stod(fields[1])),
                       std::stod(fields[3]));
    } else if (fields[0] == "reachability") {
      ASSERT_EQ(fields.size(), 5U);
      std::vector<std::pair<std::uint64_t, std::uint64_t>> edges;
      for (const auto& edge : Split(fields[1], ',')) {
        const auto endpoints = Split(edge, '>');
        ASSERT_EQ(endpoints.size(), 2U);
        edges.emplace_back(std::stoull(endpoints[0]), std::stoull(endpoints[1]));
      }
      EXPECT_EQ(junctionlens::eval::ReachableWithinHops(edges, std::stoull(fields[2]),
                                                        std::stoull(fields[3]), 3U),
                fields[4] == "1");
    } else if (fields[0] == "flip") {
      ASSERT_EQ(fields.size(), 5U);
      std::vector<bool> states;
      for (const auto& state : Split(fields[1], ',')) {
        states.push_back(state == "1");
      }
      const auto result = junctionlens::eval::StateFlipRate(states);
      ASSERT_TRUE(result.value.has_value());
      EXPECT_DOUBLE_EQ(result.numerator, std::stod(fields[2]));
      EXPECT_DOUBLE_EQ(result.denominator, std::stod(fields[3]));
      EXPECT_DOUBLE_EQ(*result.value, std::stod(fields[4]));
    } else if (fields[0] == "endpoint_gap") {
      ASSERT_EQ(fields.size(), 4U);
      const auto source_values = Doubles(fields[1]);
      const auto target_values = Doubles(fields[2]);
      ASSERT_EQ(source_values.size(), 3U);
      ASSERT_EQ(target_values.size(), 3U);
      const std::array<double, 3> source_endpoint{source_values[0], source_values[1],
                                                  source_values[2]};
      const std::array<double, 3> target_endpoint{target_values[0], target_values[1],
                                                  target_values[2]};
      EXPECT_DOUBLE_EQ(junctionlens::eval::EndpointGap(source_endpoint, target_endpoint),
                       std::stod(fields[3]));
    } else {
      FAIL() << "unsupported language-neutral golden: " << fields[0];
    }
    ++cases;
  }
  EXPECT_EQ(cases, 7U);
}

TEST(EvalMetrics, RejectsInvalidNumericInputs) {
  EXPECT_THROW((void)junctionlens::eval::SafeRatio(2.0, 1.0), std::invalid_argument);
  EXPECT_THROW((void)junctionlens::eval::LinearQuantile({}, 0.5), std::invalid_argument);
  EXPECT_FALSE(junctionlens::eval::StateFlipRate({true}).value.has_value());
}

TEST(EvalMetrics, CalibrationAndCoverageMatchAnalyticGoldens) {
  const std::vector<double> probabilities{0.0, 0.25, 0.75, 1.0};
  const std::vector<std::uint8_t> outcomes{0U, 0U, 1U, 1U};
  EXPECT_DOUBLE_EQ(junctionlens::eval::BinaryBrier(probabilities, outcomes), 0.03125);
  const auto nll = junctionlens::eval::BinaryNll(probabilities, outcomes);
  EXPECT_EQ(nll.saturation_count, 2U);
  const double expected_nll = (-2.0 * std::log(1.0 - 1.0e-7) - 2.0 * std::log(0.75)) / 4.0;
  EXPECT_NEAR(nll.value, expected_nll, 1.0e-15);
  const auto coverage = junctionlens::eval::MarginalLaplaceCoverage90(
      {std::log(10.0), std::log(10.0) + 0.001}, {1.0, 1.0}, {1.0, 1.0});
  ASSERT_TRUE(coverage.value.has_value());
  EXPECT_DOUBLE_EQ(*coverage.value, 0.5);
}

}  // namespace
