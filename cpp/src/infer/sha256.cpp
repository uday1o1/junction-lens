#include "junctionlens/infer/runtime.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <string>

namespace junctionlens::infer {
namespace {

constexpr std::array<std::uint32_t, 64> kRoundConstants = {
    0x428a2f98U, 0x71374491U, 0xb5c0fbcfU, 0xe9b5dba5U, 0x3956c25bU, 0x59f111f1U, 0x923f82a4U,
    0xab1c5ed5U, 0xd807aa98U, 0x12835b01U, 0x243185beU, 0x550c7dc3U, 0x72be5d74U, 0x80deb1feU,
    0x9bdc06a7U, 0xc19bf174U, 0xe49b69c1U, 0xefbe4786U, 0x0fc19dc6U, 0x240ca1ccU, 0x2de92c6fU,
    0x4a7484aaU, 0x5cb0a9dcU, 0x76f988daU, 0x983e5152U, 0xa831c66dU, 0xb00327c8U, 0xbf597fc7U,
    0xc6e00bf3U, 0xd5a79147U, 0x06ca6351U, 0x14292967U, 0x27b70a85U, 0x2e1b2138U, 0x4d2c6dfcU,
    0x53380d13U, 0x650a7354U, 0x766a0abbU, 0x81c2c92eU, 0x92722c85U, 0xa2bfe8a1U, 0xa81a664bU,
    0xc24b8b70U, 0xc76c51a3U, 0xd192e819U, 0xd6990624U, 0xf40e3585U, 0x106aa070U, 0x19a4c116U,
    0x1e376c08U, 0x2748774cU, 0x34b0bcb5U, 0x391c0cb3U, 0x4ed8aa4aU, 0x5b9cca4fU, 0x682e6ff3U,
    0x748f82eeU, 0x78a5636fU, 0x84c87814U, 0x8cc70208U, 0x90befffaU, 0xa4506cebU, 0xbef9a3f7U,
    0xc67178f2U,
};

class Sha256 final {
 public:
  void Update(const std::uint8_t* bytes, std::size_t size) {
    if (byte_count_ > std::numeric_limits<std::uint64_t>::max() / 8U ||
        size > std::numeric_limits<std::uint64_t>::max() / 8U - byte_count_) {
      throw RuntimeError("RUNTIME_SHA256_SIZE", "SHA-256 input is too large");
    }
    byte_count_ += static_cast<std::uint64_t>(size);
    while (size > 0U) {
      const std::size_t copied = std::min(size, block_.size() - block_size_);
      std::copy_n(bytes, copied, block_.begin() + static_cast<std::ptrdiff_t>(block_size_));
      bytes += copied;
      size -= copied;
      block_size_ += copied;
      if (block_size_ == block_.size()) {
        Transform();
        block_size_ = 0U;
      }
    }
  }

  [[nodiscard]] std::array<std::uint8_t, 32> Finalize() {
    const std::uint64_t bit_count = byte_count_ * 8U;
    block_[block_size_++] = 0x80U;
    if (block_size_ > 56U) {
      std::fill(block_.begin() + static_cast<std::ptrdiff_t>(block_size_), block_.end(), 0U);
      Transform();
      block_size_ = 0U;
    }
    std::fill(block_.begin() + static_cast<std::ptrdiff_t>(block_size_), block_.begin() + 56, 0U);
    for (std::size_t index = 0; index < 8U; ++index) {
      block_[63U - index] = static_cast<std::uint8_t>(bit_count >> (index * 8U));
    }
    Transform();
    std::array<std::uint8_t, 32> digest{};
    for (std::size_t word = 0; word < state_.size(); ++word) {
      for (std::size_t byte = 0; byte < 4U; ++byte) {
        digest[word * 4U + byte] = static_cast<std::uint8_t>(state_[word] >> ((3U - byte) * 8U));
      }
    }
    return digest;
  }

 private:
  void Transform() {
    std::array<std::uint32_t, 64> schedule{};
    for (std::size_t index = 0; index < 16U; ++index) {
      const std::size_t offset = index * 4U;
      schedule[index] = (static_cast<std::uint32_t>(block_[offset]) << 24U) |
                        (static_cast<std::uint32_t>(block_[offset + 1U]) << 16U) |
                        (static_cast<std::uint32_t>(block_[offset + 2U]) << 8U) |
                        static_cast<std::uint32_t>(block_[offset + 3U]);
    }
    for (std::size_t index = 16U; index < schedule.size(); ++index) {
      const std::uint32_t s0 = std::rotr(schedule[index - 15U], 7) ^
                               std::rotr(schedule[index - 15U], 18) ^ (schedule[index - 15U] >> 3U);
      const std::uint32_t s1 = std::rotr(schedule[index - 2U], 17) ^
                               std::rotr(schedule[index - 2U], 19) ^ (schedule[index - 2U] >> 10U);
      schedule[index] = schedule[index - 16U] + s0 + schedule[index - 7U] + s1;
    }
    std::uint32_t a = state_[0];
    std::uint32_t b = state_[1];
    std::uint32_t c = state_[2];
    std::uint32_t d = state_[3];
    std::uint32_t e = state_[4];
    std::uint32_t f = state_[5];
    std::uint32_t g = state_[6];
    std::uint32_t h = state_[7];
    for (std::size_t index = 0; index < schedule.size(); ++index) {
      const std::uint32_t sum1 = std::rotr(e, 6) ^ std::rotr(e, 11) ^ std::rotr(e, 25);
      const std::uint32_t choice = (e & f) ^ (~e & g);
      const std::uint32_t temporary1 = h + sum1 + choice + kRoundConstants[index] + schedule[index];
      const std::uint32_t sum0 = std::rotr(a, 2) ^ std::rotr(a, 13) ^ std::rotr(a, 22);
      const std::uint32_t majority = (a & b) ^ (a & c) ^ (b & c);
      const std::uint32_t temporary2 = sum0 + majority;
      h = g;
      g = f;
      f = e;
      e = d + temporary1;
      d = c;
      c = b;
      b = a;
      a = temporary1 + temporary2;
    }
    state_[0] += a;
    state_[1] += b;
    state_[2] += c;
    state_[3] += d;
    state_[4] += e;
    state_[5] += f;
    state_[6] += g;
    state_[7] += h;
  }

  std::array<std::uint32_t, 8> state_ = {0x6a09e667U, 0xbb67ae85U, 0x3c6ef372U, 0xa54ff53aU,
                                         0x510e527fU, 0x9b05688cU, 0x1f83d9abU, 0x5be0cd19U};
  std::array<std::uint8_t, 64> block_{};
  std::size_t block_size_ = 0U;
  std::uint64_t byte_count_ = 0U;
};

[[nodiscard]] std::string HexDigest(const std::array<std::uint8_t, 32>& digest) {
  std::ostringstream output;
  output << std::hex << std::setfill('0');
  for (const std::uint8_t byte : digest) {
    output << std::setw(2) << static_cast<unsigned int>(byte);
  }
  return output.str();
}

}  // namespace

std::string Sha256File(const std::filesystem::path& path) {
  std::ifstream input(path, std::ios::binary);
  if (!input) {
    throw RuntimeError("RUNTIME_MODEL_IO",
                       "file could not be opened for hashing: " + path.string());
  }
  Sha256 hasher;
  std::array<std::uint8_t, 64U * 1024U> buffer{};
  while (input) {
    input.read(reinterpret_cast<char*>(buffer.data()), static_cast<std::streamsize>(buffer.size()));
    const auto count = input.gcount();
    if (count > 0) {
      hasher.Update(buffer.data(), static_cast<std::size_t>(count));
    }
  }
  if (!input.eof()) {
    throw RuntimeError("RUNTIME_MODEL_IO", "file could not be read for hashing: " + path.string());
  }
  return HexDigest(hasher.Finalize());
}

std::string Sha256Text(const std::string_view value) {
  Sha256 hasher;
  hasher.Update(reinterpret_cast<const std::uint8_t*>(value.data()), value.size());
  return HexDigest(hasher.Finalize());
}

}  // namespace junctionlens::infer
