#pragma once

#include <cstddef>
#include <filesystem>
#include <string>

#include "junctionlens/v1/scene_control_graph.pb.h"

namespace junctionlens::contract {

inline constexpr std::size_t kMaximumSerializedBytes = 64U * 1024U * 1024U;

struct ValidationResult {
  bool valid;
  std::string reason_code;
  std::string path;
  std::string detail;
};

[[nodiscard]] ValidationResult Validate(const v1::SceneControlGraphEnvelope& envelope);

[[nodiscard]] ValidationResult ParseFile(
    const std::filesystem::path& path,
    v1::SceneControlGraphEnvelope& envelope
);

void VerifyExactProtobufRuntime();

}  // namespace junctionlens::contract
