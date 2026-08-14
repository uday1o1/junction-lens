#include <cstddef>
#include <cstdint>
#include <limits>

#include "contract/validation.hpp"

extern "C" int LLVMFuzzerTestOneInput(const std::uint8_t* data, std::size_t size) {
  if (size > junctionlens::contract::kMaximumSerializedBytes ||
      size > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
    return 0;
  }
  junctionlens::v1::SceneControlGraphEnvelope envelope;
  if (envelope.ParseFromArray(data, static_cast<int>(size))) {
    static_cast<void>(junctionlens::contract::Validate(envelope));
  }
  return 0;
}
