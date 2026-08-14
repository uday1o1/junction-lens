#include "junctionlens/infer/runtime.hpp"

#include <algorithm>
#include <utility>

namespace junctionlens::infer {
namespace {

[[nodiscard]] BufferState ExpectedNext(const BufferState state) {
  switch (state) {
    case BufferState::kDecoding:
      return BufferState::kPreprocessing;
    case BufferState::kPreprocessing:
      return BufferState::kInference;
    case BufferState::kInference:
      return BufferState::kPostprocessing;
    case BufferState::kPostprocessing:
      return BufferState::kSerializing;
    case BufferState::kFree:
    case BufferState::kSerializing:
      break;
  }
  throw RuntimeError("RUNTIME_BUFFER_TRANSITION", "buffer has no normal successor state");
}

}  // namespace

RuntimeError::RuntimeError(std::string reason_code, std::string detail)
    : std::runtime_error(std::move(detail)), reason_code_(std::move(reason_code)) {}

const std::string& RuntimeError::reason_code() const noexcept { return reason_code_; }

std::string_view BufferStateName(const BufferState state) noexcept {
  switch (state) {
    case BufferState::kFree:
      return "FREE";
    case BufferState::kDecoding:
      return "DECODING";
    case BufferState::kPreprocessing:
      return "PREPROCESSING";
    case BufferState::kInference:
      return "INFERENCE";
    case BufferState::kPostprocessing:
      return "POSTPROCESSING";
    case BufferState::kSerializing:
      return "SERIALIZING";
  }
  return "UNKNOWN";
}

BufferLease::BufferLease(BufferPool* pool, const std::size_t slot_index) noexcept
    : pool_(pool), slot_index_(slot_index) {}

BufferLease::BufferLease(BufferLease&& other) noexcept
    : pool_(std::exchange(other.pool_, nullptr)), slot_index_(other.slot_index_) {}

BufferLease& BufferLease::operator=(BufferLease&& other) noexcept {
  if (this != &other) {
    Release();
    pool_ = std::exchange(other.pool_, nullptr);
    slot_index_ = other.slot_index_;
  }
  return *this;
}

BufferLease::~BufferLease() { Release(); }

void BufferLease::Advance(const BufferState next) {
  if (pool_ == nullptr) {
    throw RuntimeError("RUNTIME_BUFFER_RELEASED", "cannot advance a released buffer lease");
  }
  pool_->Advance(slot_index_, next);
}

void BufferLease::Release() {
  if (pool_ != nullptr) {
    pool_->Release(slot_index_);
    pool_ = nullptr;
  }
}

std::size_t BufferLease::slot_index() const {
  if (pool_ == nullptr) {
    throw RuntimeError("RUNTIME_BUFFER_RELEASED", "buffer lease has already been released");
  }
  return slot_index_;
}

BufferState BufferLease::state() const {
  if (pool_ == nullptr) {
    throw RuntimeError("RUNTIME_BUFFER_RELEASED", "buffer lease has already been released");
  }
  return pool_->state(slot_index_);
}

BufferPool::BufferPool(const std::size_t capacity) : states_(capacity, BufferState::kFree) {
  if (capacity == 0U || capacity > 1024U) {
    throw RuntimeError("RUNTIME_BUFFER_CAPACITY", "buffer capacity must be within [1, 1024]");
  }
}

BufferLease BufferPool::Acquire() {
  const auto found = std::find(states_.begin(), states_.end(), BufferState::kFree);
  if (found == states_.end()) {
    throw RuntimeError("RUNTIME_BUFFER_EXHAUSTED", "offline buffer pool is full");
  }
  const auto index = static_cast<std::size_t>(std::distance(states_.begin(), found));
  states_[index] = BufferState::kDecoding;
  ++current_depth_;
  high_water_mark_ = std::max(high_water_mark_, current_depth_);
  return BufferLease(this, index);
}

std::size_t BufferPool::capacity() const noexcept { return states_.size(); }

std::size_t BufferPool::current_depth() const noexcept { return current_depth_; }

std::size_t BufferPool::high_water_mark() const noexcept { return high_water_mark_; }

bool BufferPool::all_free() const noexcept {
  return current_depth_ == 0U &&
         std::all_of(states_.begin(), states_.end(),
                     [](const BufferState state) { return state == BufferState::kFree; });
}

BufferState BufferPool::state(const std::size_t slot_index) const {
  if (slot_index >= states_.size()) {
    throw RuntimeError("RUNTIME_BUFFER_INDEX", "buffer slot index is outside the pool");
  }
  return states_[slot_index];
}

void BufferPool::Advance(const std::size_t slot_index, const BufferState next) {
  const BufferState current = state(slot_index);
  if (current == BufferState::kFree || current == BufferState::kSerializing ||
      ExpectedNext(current) != next) {
    throw RuntimeError("RUNTIME_BUFFER_TRANSITION",
                       "invalid buffer transition from " + std::string(BufferStateName(current)) +
                           " to " + std::string(BufferStateName(next)));
  }
  states_[slot_index] = next;
}

void BufferPool::Release(const std::size_t slot_index) noexcept {
  if (slot_index < states_.size() && states_[slot_index] != BufferState::kFree) {
    states_[slot_index] = BufferState::kFree;
    if (current_depth_ > 0U) {
      --current_depth_;
    }
  }
}

}  // namespace junctionlens::infer
