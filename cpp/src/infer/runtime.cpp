#include "junctionlens/infer/runtime.hpp"

#include <cpu_provider_factory.h>
#include <onnxruntime_cxx_api.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <map>
#include <numeric>
#include <span>
#include <string>
#include <string_view>
#include <tuple>
#include <utility>
#include <vector>

#include "contract/validation.hpp"

namespace junctionlens::infer {
namespace {

constexpr std::string_view kInputContractSha256 =
    "cf5adc1545fa9b82f2a4429adab5c020bc64bd0357c908e5457f88dd62ea34ef";
constexpr std::string_view kOutputContractSha256 =
    "84081fbd524a0439d2ec5f6c26fb33baf72cea11dceefd85241f7f4750b5f495";
constexpr std::string_view kCpuProviderDigestInput =
    "onnxruntime-1.25.0|CPUExecutionProvider|all-model-nodes";

struct TensorSpec {
  std::string_view name;
  ONNXTensorElementDataType element_type;
  std::vector<std::int64_t> dimensions;
};

[[nodiscard]] const std::vector<TensorSpec>& InputSpecs() {
  static const std::vector<TensorSpec> specs = {
      {"images", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, {-1, 2, 8, 3, 384, 640}},
      {"camera_valid", ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL, {-1, 2, 8}},
      {"intrinsics", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, {-1, 2, 8, 3, 3}},
      {"t_vehicle_camera", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, {-1, 2, 8, 4, 4}},
      {"ego_motion_previous_to_current", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, {-1, 4, 4}},
      {"temporal_valid", ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL, {-1}},
  };
  return specs;
}

[[nodiscard]] const std::vector<TensorSpec>& OutputSpecs() {
  static const std::vector<TensorSpec> specs = {
      {"lane_existence_logits", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, {-1, 96}},
      {"lane_centerline", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, {-1, 96, 11, 3}},
      {"lane_left_boundary", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, {-1, 96, 11, 3}},
      {"lane_right_boundary", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, {-1, 96, 11, 3}},
      {"lane_left_boundary_logits", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, {-1, 96, 3}},
      {"lane_right_boundary_logits", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, {-1, 96, 3}},
      {"lane_connector_logits", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, {-1, 96}},
      {"lane_geometry_scales", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, {-1, 96, 3, 11, 3}},
      {"lane_track_embeddings", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, {-1, 96, 16}},
      {"traffic_existence_logits", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, {-1, 64}},
      {"traffic_boxes", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, {-1, 64, 4}},
      {"traffic_category_logits", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, {-1, 64, 2}},
      {"traffic_attribute_logits", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, {-1, 64, 13}},
      {"traffic_box_scales", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, {-1, 64, 4}},
      {"traffic_track_embeddings", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, {-1, 64, 16}},
      {"area_existence_logits", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, {-1, 32}},
      {"area_category_logits", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, {-1, 32, 2}},
      {"area_points", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, {-1, 32, 20, 3}},
      {"area_valid_logits", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, {-1, 32, 20}},
      {"area_geometry_scales", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, {-1, 32, 20, 3}},
      {"area_track_embeddings", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, {-1, 32, 16}},
      {"lane_successor_logits", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, {-1, 96, 96}},
      {"control_lane_logits", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, {-1, 64, 96}},
  };
  return specs;
}

[[nodiscard]] bool IsLowerHex(const std::string_view value, const std::size_t size) {
  return value.size() == size && std::all_of(value.begin(), value.end(), [](const char character) {
           return (character >= '0' && character <= '9') || (character >= 'a' && character <= 'f');
         });
}

void ValidateOptions(const RuntimeOptions& options) {
  if (options.model_path.empty() || !std::filesystem::is_regular_file(options.model_path)) {
    throw RuntimeError("RUNTIME_MODEL_IO", "model path is absent or not a regular file");
  }
  if (!IsLowerHex(options.expected_profile_sha256, 64U)) {
    throw RuntimeError("RUNTIME_PROFILE_DIGEST",
                       "expected profile digest must be lowercase SHA-256");
  }
  if (!IsLowerHex(options.producer.git_commit, 40U)) {
    throw RuntimeError("RUNTIME_GIT_COMMIT", "producer Git commit must be a lowercase 40-byte SHA");
  }
  if (!IsLowerHex(options.producer.configuration_sha256, 64U) ||
      !IsLowerHex(options.producer.runtime_build_sha256, 64U)) {
    throw RuntimeError("RUNTIME_PRODUCER_DIGEST", "producer digests must be lowercase SHA-256");
  }
  if (!(options.node_threshold >= 0.0 && options.node_threshold <= 1.0) ||
      !(options.edge_threshold >= 0.0 && options.edge_threshold <= 1.0)) {
    throw RuntimeError("RUNTIME_THRESHOLD", "node and edge thresholds must be within [0, 1]");
  }
}

void ValidateTensorContract(const Ort::Session& session, const bool input) {
  Ort::AllocatorWithDefaultOptions allocator;
  const auto& specs = input ? InputSpecs() : OutputSpecs();
  const std::size_t observed_count = input ? session.GetInputCount() : session.GetOutputCount();
  if (observed_count != specs.size()) {
    throw RuntimeError("RUNTIME_TENSOR_COUNT", "tensor count differs from the frozen contract");
  }
  for (std::size_t index = 0; index < specs.size(); ++index) {
    auto name = input ? session.GetInputNameAllocated(index, allocator)
                      : session.GetOutputNameAllocated(index, allocator);
    if (name.get() == nullptr || std::string_view(name.get()) != specs[index].name) {
      throw RuntimeError("RUNTIME_TENSOR_NAME",
                         "tensor name differs at index " + std::to_string(index));
    }
    const Ort::TypeInfo type_info =
        input ? session.GetInputTypeInfo(index) : session.GetOutputTypeInfo(index);
    const auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
    if (tensor_info.GetElementType() != specs[index].element_type) {
      throw RuntimeError("RUNTIME_TENSOR_TYPE",
                         "tensor element type differs for " + std::string(specs[index].name));
    }
    const auto observed_shape = tensor_info.GetShape();
    const bool known_softplus_annotation =
        !input &&
        (specs[index].name == "lane_geometry_scales" ||
         specs[index].name == "area_geometry_scales") &&
        !observed_shape.empty() && observed_shape[0] == 1 &&
        std::vector<std::int64_t>(observed_shape.begin() + 1, observed_shape.end()) ==
            std::vector<std::int64_t>(specs[index].dimensions.begin() + 1,
                                      specs[index].dimensions.end());
    if (observed_shape != specs[index].dimensions && !known_softplus_annotation) {
      throw RuntimeError("RUNTIME_TENSOR_SHAPE",
                         "tensor shape differs for " + std::string(specs[index].name));
    }
  }
}

[[nodiscard]] std::map<std::string, std::string> ReadMetadata(const Ort::Session& session) {
  Ort::AllocatorWithDefaultOptions allocator;
  const Ort::ModelMetadata metadata = session.GetModelMetadata();
  std::map<std::string, std::string> result;
  for (auto& key : metadata.GetCustomMetadataMapKeysAllocated(allocator)) {
    if (key.get() != nullptr) {
      auto value = metadata.LookupCustomMetadataMapAllocated(key.get(), allocator);
      if (value.get() != nullptr) {
        result.emplace(key.get(), value.get());
      }
    }
  }
  return result;
}

void RequireMetadata(const std::map<std::string, std::string>& metadata, const std::string_view key,
                     const std::string_view expected) {
  const auto found = metadata.find(std::string(key));
  if (found == metadata.end() || found->second != expected) {
    throw RuntimeError("RUNTIME_MODEL_METADATA",
                       "required model metadata differs: " + std::string(key));
  }
}

void ValidateMetadata(const std::map<std::string, std::string>& metadata,
                      const RuntimeOptions& options) {
  RequireMetadata(metadata, "junctionlens.schema_version", "1.0.0");
  RequireMetadata(metadata, "junctionlens.profile_sha256", options.expected_profile_sha256);
  RequireMetadata(metadata, "junctionlens.input_contract_sha256", kInputContractSha256);
  RequireMetadata(metadata, "junctionlens.output_contract_sha256", kOutputContractSha256);
  RequireMetadata(metadata, "junctionlens.opset", "18");
  RequireMetadata(metadata, "junctionlens.precision", "fp32");
  const auto profile = metadata.find("junctionlens.profile_id");
  const auto checkpoint = metadata.find("junctionlens.checkpoint_sha256");
  if (profile == metadata.end() || profile->second.empty() || checkpoint == metadata.end() ||
      !IsLowerHex(checkpoint->second, 64U)) {
    throw RuntimeError("RUNTIME_MODEL_METADATA", "profile or checkpoint metadata is malformed");
  }
}

template <typename Element>
[[nodiscard]] Ort::Value MakeTensor(Ort::MemoryInfo& memory, const std::vector<Element>& values,
                                    const std::vector<std::int64_t>& shape) {
  return Ort::Value::CreateTensor<Element>(memory, const_cast<Element*>(values.data()),
                                           values.size(), shape.data(), shape.size());
}

[[nodiscard]] Ort::Value MakeBoolTensor(Ort::MemoryInfo& memory,
                                        const std::vector<std::uint8_t>& values,
                                        const std::vector<std::int64_t>& shape) {
  return Ort::Value::CreateTensor(memory, const_cast<std::uint8_t*>(values.data()), values.size(),
                                  shape.data(), shape.size(), ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL);
}

[[nodiscard]] double Sigmoid(const float value) {
  const double number = static_cast<double>(value);
  if (number >= 0.0) {
    const double exponent = std::exp(-number);
    return 1.0 / (1.0 + exponent);
  }
  const double exponent = std::exp(number);
  return exponent / (1.0 + exponent);
}

[[nodiscard]] std::vector<double> Softmax(const float* values, const std::size_t count) {
  const float maximum = *std::max_element(values, values + count);
  std::vector<double> result(count, 0.0);
  double total = 0.0;
  for (std::size_t index = 0; index < count; ++index) {
    result[index] = std::exp(static_cast<double>(values[index] - maximum));
    total += result[index];
  }
  for (double& value : result) {
    value /= total;
  }
  return result;
}

struct OutputView {
  const float* data;
  std::size_t count;
};

[[nodiscard]] OutputView View(const std::vector<Ort::Value>& outputs, const std::size_t index) {
  const auto info = outputs[index].GetTensorTypeAndShapeInfo();
  if (info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
    throw RuntimeError("RUNTIME_OUTPUT_TYPE", "model output is not float32");
  }
  const std::size_t count = info.GetElementCount();
  const float* data = outputs[index].GetTensorData<float>();
  for (std::size_t element = 0; element < count; ++element) {
    if (!std::isfinite(data[element])) {
      throw RuntimeError("RUNTIME_NONFINITE_OUTPUT", "model output contains a nonfinite value");
    }
  }
  return {data, count};
}

[[nodiscard]] std::int64_t Quantize(const float value, const double step) {
  const double scaled = static_cast<double>(value) / step;
  if (scaled < static_cast<double>(std::numeric_limits<std::int64_t>::min()) ||
      scaled > static_cast<double>(std::numeric_limits<std::int64_t>::max())) {
    throw RuntimeError("RUNTIME_GEOMETRY_RANGE", "geometry coordinate exceeds quantized range");
  }
  return static_cast<std::int64_t>(std::llround(scaled));
}

struct NodeOrder {
  std::size_t query;
  double confidence;
  std::vector<std::int64_t> geometry_key;
};

void SortNodes(std::vector<NodeOrder>& nodes) {
  std::sort(nodes.begin(), nodes.end(), [](const NodeOrder& first, const NodeOrder& second) {
    if (first.confidence != second.confidence) {
      return first.confidence > second.confidence;
    }
    if (first.geometry_key != second.geometry_key) {
      return first.geometry_key < second.geometry_key;
    }
    return first.query < second.query;
  });
}

[[nodiscard]] std::uint64_t PredictedNodeId(const v1::NodeType type, const std::size_t ordinal) {
  return (static_cast<std::uint64_t>(type) << 56U) | (static_cast<std::uint64_t>(ordinal) + 1U);
}

void AppendUint32(std::string& output, const std::uint32_t value) {
  for (int shift = 24; shift >= 0; shift -= 8) {
    output.push_back(static_cast<char>((value >> static_cast<unsigned int>(shift)) & 0xffU));
  }
}

void AppendUint64(std::string& output, const std::uint64_t value) {
  for (int shift = 56; shift >= 0; shift -= 8) {
    output.push_back(static_cast<char>((value >> static_cast<unsigned int>(shift)) & 0xffU));
  }
}

void AppendPart(std::string& output, const std::string_view value) {
  AppendUint32(output, static_cast<std::uint32_t>(value.size()));
  output.append(value);
}

[[nodiscard]] std::uint64_t EdgeId(const v1::FrameKey& frame_key, const v1::GraphEdgeType type,
                                   const std::uint64_t source, const std::uint64_t target) {
  std::string payload;
  AppendUint32(payload, 1U);
  AppendPart(payload, "junctionlens-edge-id-v1");
  AppendPart(payload, frame_key.dataset_id());
  AppendPart(payload, frame_key.dataset_version());
  AppendPart(payload, frame_key.split_id());
  AppendPart(payload, frame_key.segment_id());
  std::string timestamp;
  AppendUint64(timestamp, static_cast<std::uint64_t>(frame_key.timestamp_ns()));
  AppendPart(payload, timestamp);
  std::string domain;
  AppendUint32(domain, static_cast<std::uint32_t>(frame_key.source_domain()));
  AppendPart(payload, domain);
  AppendPart(payload, frame_key.calibration_sha256());
  AppendPart(payload, frame_key.frame_manifest_sha256());
  std::string edge;
  AppendUint32(edge, static_cast<std::uint32_t>(type));
  AppendUint64(edge, source);
  AppendUint64(edge, target);
  AppendPart(payload, edge);
  const std::string digest = Sha256Text(payload);
  const std::uint64_t value = std::stoull(digest.substr(0U, 16U), nullptr, 16);
  return value == 0U ? 1U : value;
}

void AddPoint(v1::Polyline3d& line, const float* values, const std::size_t point) {
  auto* output = line.add_points();
  output->set_x(static_cast<double>(values[point * 3U]));
  output->set_y(static_cast<double>(values[point * 3U + 1U]));
  output->set_z(static_cast<double>(values[point * 3U + 2U]));
}

void AddScale(google::protobuf::RepeatedPtrField<v1::LaplaceScale3d>* output, const float* values,
              const std::size_t point) {
  auto* scale = output->Add();
  scale->set_x(static_cast<double>(values[point * 3U]));
  scale->set_y(static_cast<double>(values[point * 3U + 1U]));
  scale->set_z(static_cast<double>(values[point * 3U + 2U]));
}

void FillDistribution(v1::ClassDistribution& output, const float* values, const std::size_t count) {
  for (const double probability : Softmax(values, count)) {
    output.add_probabilities(probability);
  }
}

[[nodiscard]] std::vector<std::size_t> AreaPointIndices(const float* logits) {
  std::vector<std::size_t> result;
  for (std::size_t index = 0; index < 20U; ++index) {
    if (Sigmoid(logits[index]) >= 0.5) {
      result.push_back(index);
    }
  }
  if (result.size() < 2U) {
    std::array<std::size_t, 20> indices{};
    std::iota(indices.begin(), indices.end(), 0U);
    std::partial_sort(indices.begin(), indices.begin() + 2, indices.end(),
                      [logits](const std::size_t first, const std::size_t second) {
                        if (logits[first] != logits[second]) {
                          return logits[first] > logits[second];
                        }
                        return first < second;
                      });
    result = {indices[0], indices[1]};
    std::sort(result.begin(), result.end());
  }
  return result;
}

[[nodiscard]] std::array<double, 4> OrderedBox(const float* values) {
  double x_min = std::min(static_cast<double>(values[0]), static_cast<double>(values[2]));
  double x_max = std::max(static_cast<double>(values[0]), static_cast<double>(values[2]));
  double y_min = std::min(static_cast<double>(values[1]), static_cast<double>(values[3]));
  double y_max = std::max(static_cast<double>(values[1]), static_cast<double>(values[3]));
  x_min = std::clamp(x_min, 0.0, 1.0);
  x_max = std::clamp(x_max, 0.0, 1.0);
  y_min = std::clamp(y_min, 0.0, 1.0);
  y_max = std::clamp(y_max, 0.0, 1.0);
  if (x_min == x_max) {
    x_min = std::max(0.0, x_min - 1.0e-7);
    x_max = std::min(1.0, x_max + 1.0e-7);
  }
  if (y_min == y_max) {
    y_min = std::max(0.0, y_min - 1.0e-7);
    y_max = std::min(1.0, y_max + 1.0e-7);
  }
  return {x_min, y_min, x_max, y_max};
}

struct PostprocessMaps {
  std::array<std::uint64_t, 96> lane{};
  std::array<std::uint64_t, 64> traffic{};
};

[[nodiscard]] PostprocessMaps AddNodes(const std::vector<Ort::Value>& outputs,
                                       const RuntimeOptions& options,
                                       v1::SceneControlGraph& graph) {
  const OutputView lane_existence = View(outputs, 0U);
  const OutputView lane_centerline = View(outputs, 1U);
  const OutputView lane_left = View(outputs, 2U);
  const OutputView lane_right = View(outputs, 3U);
  const OutputView lane_left_type = View(outputs, 4U);
  const OutputView lane_right_type = View(outputs, 5U);
  const OutputView lane_connector = View(outputs, 6U);
  const OutputView lane_scales = View(outputs, 7U);
  const OutputView traffic_existence = View(outputs, 9U);
  const OutputView traffic_boxes = View(outputs, 10U);
  const OutputView traffic_category = View(outputs, 11U);
  const OutputView traffic_attribute = View(outputs, 12U);
  const OutputView area_existence = View(outputs, 15U);
  const OutputView area_category = View(outputs, 16U);
  const OutputView area_points = View(outputs, 17U);
  const OutputView area_valid = View(outputs, 18U);
  const OutputView area_scales = View(outputs, 19U);
  std::vector<NodeOrder> lanes;
  std::vector<NodeOrder> controls;
  std::vector<NodeOrder> areas;
  std::array<std::vector<std::size_t>, 32> area_indices;
  for (std::size_t query = 0; query < 96U; ++query) {
    const double confidence = Sigmoid(lane_existence.data[query]);
    if (confidence >= options.node_threshold) {
      std::vector<std::int64_t> key;
      key.reserve(99U);
      for (const OutputView geometry : {lane_centerline, lane_left, lane_right}) {
        const float* values = geometry.data + query * 33U;
        for (std::size_t index = 0; index < 33U; ++index) {
          key.push_back(Quantize(values[index], 0.001));
        }
      }
      lanes.push_back({query, confidence, std::move(key)});
    }
  }
  for (std::size_t query = 0; query < 64U; ++query) {
    const double confidence = Sigmoid(traffic_existence.data[query]);
    if (confidence >= options.node_threshold) {
      std::vector<std::int64_t> key;
      key.reserve(4U);
      const auto box = OrderedBox(traffic_boxes.data + query * 4U);
      for (const double value : box) {
        key.push_back(static_cast<std::int64_t>(std::llround(value / 0.000001)));
      }
      controls.push_back({query, confidence, std::move(key)});
    }
  }
  for (std::size_t query = 0; query < 32U; ++query) {
    const double confidence = Sigmoid(area_existence.data[query]);
    if (confidence >= options.node_threshold) {
      area_indices[query] = AreaPointIndices(area_valid.data + query * 20U);
      std::vector<std::int64_t> key;
      key.reserve(area_indices[query].size() * 3U);
      for (const std::size_t point : area_indices[query]) {
        for (std::size_t coordinate = 0; coordinate < 3U; ++coordinate) {
          key.push_back(Quantize(area_points.data[query * 60U + point * 3U + coordinate], 0.001));
        }
      }
      areas.push_back({query, confidence, std::move(key)});
    }
  }
  SortNodes(lanes);
  SortNodes(controls);
  SortNodes(areas);
  PostprocessMaps maps;
  for (std::size_t ordinal = 0; ordinal < lanes.size(); ++ordinal) {
    const NodeOrder& order = lanes[ordinal];
    const std::uint64_t node_id = PredictedNodeId(v1::NODE_TYPE_LANE_SEGMENT, ordinal);
    maps.lane[order.query] = node_id;
    auto* lane = graph.add_lanes();
    lane->set_node_id(node_id);
    lane->set_decoder_query_index(static_cast<std::uint32_t>(order.query));
    lane->set_existence_confidence(order.confidence);
    lane->set_intersection_or_connector_probability(Sigmoid(lane_connector.data[order.query]));
    const std::array<OutputView, 3> geometry = {lane_centerline, lane_left, lane_right};
    const std::array<v1::Polyline3d*, 3> polylines = {
        lane->mutable_centerline(), lane->mutable_left_boundary(), lane->mutable_right_boundary()};
    for (std::size_t kind = 0; kind < geometry.size(); ++kind) {
      const float* points = geometry[kind].data + order.query * 33U;
      const float* scales = lane_scales.data + order.query * 99U + kind * 33U;
      polylines[kind]->set_confidence(order.confidence);
      for (std::size_t point = 0; point < 11U; ++point) {
        AddPoint(*polylines[kind], points, point);
        AddScale(polylines[kind]->mutable_point_uncertainty(), scales, point);
        if (kind == 0U) {
          AddScale(lane->mutable_centerline_laplace_scale_m(), scales, point);
        }
      }
    }
    FillDistribution(*lane->mutable_left_boundary_type(), lane_left_type.data + order.query * 3U,
                     3U);
    FillDistribution(*lane->mutable_right_boundary_type(), lane_right_type.data + order.query * 3U,
                     3U);
  }
  for (std::size_t ordinal = 0; ordinal < controls.size(); ++ordinal) {
    const NodeOrder& order = controls[ordinal];
    const std::uint64_t node_id = PredictedNodeId(v1::NODE_TYPE_TRAFFIC_CONTROL, ordinal);
    maps.traffic[order.query] = node_id;
    auto* control = graph.add_traffic_controls();
    control->set_node_id(node_id);
    control->set_decoder_query_index(static_cast<std::uint32_t>(order.query));
    control->set_source_camera(v1::CAMERA_SLOT_FRONT_CENTER);
    control->set_existence_confidence(order.confidence);
    const auto box = OrderedBox(traffic_boxes.data + order.query * 4U);
    auto* output_box = control->mutable_normalized_half_open_box();
    output_box->set_x_min(box[0]);
    output_box->set_y_min(box[1]);
    output_box->set_x_max(box[2]);
    output_box->set_y_max(box[3]);
    FillDistribution(*control->mutable_category_distribution(),
                     traffic_category.data + order.query * 2U, 2U);
    FillDistribution(*control->mutable_attribute_distribution(),
                     traffic_attribute.data + order.query * 13U, 13U);
    control->set_calibrated_class_confidence(
        *std::max_element(control->category_distribution().probabilities().begin(),
                          control->category_distribution().probabilities().end()));
    control->set_calibrated_attribute_confidence(
        *std::max_element(control->attribute_distribution().probabilities().begin(),
                          control->attribute_distribution().probabilities().end()));
  }
  for (std::size_t ordinal = 0; ordinal < areas.size(); ++ordinal) {
    const NodeOrder& order = areas[ordinal];
    auto* area = graph.add_road_areas();
    area->set_node_id(PredictedNodeId(v1::NODE_TYPE_ROAD_AREA, ordinal));
    area->set_decoder_query_index(static_cast<std::uint32_t>(order.query));
    area->set_existence_confidence(order.confidence);
    FillDistribution(*area->mutable_category_distribution(), area_category.data + order.query * 2U,
                     2U);
    area->mutable_geometry()->set_confidence(order.confidence);
    for (const std::size_t point : area_indices[order.query]) {
      AddPoint(*area->mutable_geometry(), area_points.data + order.query * 60U, point);
      AddScale(area->mutable_geometry()->mutable_point_uncertainty(),
               area_scales.data + order.query * 60U, point);
      AddScale(area->mutable_geometry_uncertainty(), area_scales.data + order.query * 60U, point);
    }
  }
  return maps;
}

void AddEdges(const std::vector<Ort::Value>& outputs, const RuntimeOptions& options,
              const PostprocessMaps& maps, v1::SceneControlGraph& graph) {
  struct EdgeRecord {
    v1::GraphEdgeType type;
    std::uint64_t source;
    std::uint64_t target;
    double probability;
  };
  const OutputView successor = View(outputs, 21U);
  const OutputView control_lane = View(outputs, 22U);
  std::vector<EdgeRecord> edges;
  for (std::size_t source = 0; source < maps.lane.size(); ++source) {
    if (maps.lane[source] == 0U) {
      continue;
    }
    for (std::size_t target = 0; target < maps.lane.size(); ++target) {
      if (maps.lane[target] == 0U) {
        continue;
      }
      const double probability = Sigmoid(successor.data[source * 96U + target]);
      if (probability >= options.edge_threshold) {
        edges.push_back({v1::GRAPH_EDGE_TYPE_LANE_SUCCESSOR, maps.lane[source], maps.lane[target],
                         probability});
      }
    }
  }
  for (std::size_t control = 0; control < maps.traffic.size(); ++control) {
    if (maps.traffic[control] == 0U) {
      continue;
    }
    for (std::size_t lane = 0; lane < maps.lane.size(); ++lane) {
      if (maps.lane[lane] == 0U) {
        continue;
      }
      const double probability = Sigmoid(control_lane.data[control * 96U + lane]);
      if (probability >= options.edge_threshold) {
        edges.push_back({v1::GRAPH_EDGE_TYPE_CONTROL_APPLIES_TO_LANE, maps.traffic[control],
                         maps.lane[lane], probability});
      }
    }
  }
  std::sort(edges.begin(), edges.end(), [](const EdgeRecord& first, const EdgeRecord& second) {
    return std::tie(first.type, first.source, first.target) <
           std::tie(second.type, second.source, second.target);
  });
  for (const EdgeRecord& record : edges) {
    auto* edge = graph.add_edges();
    edge->set_edge_type(record.type);
    edge->set_source_node_id(record.source);
    edge->set_target_node_id(record.target);
    edge->set_raw_probability(record.probability);
    edge->set_calibrated_probability(record.probability);
    edge->set_binary_decision(true);
    edge->set_edge_id(EdgeId(graph.frame_key(), record.type, record.source, record.target));
  }
}

[[nodiscard]] v1::SceneControlGraphEnvelope Postprocess(const std::vector<Ort::Value>& outputs,
                                                        const PreprocessedInputs& inputs,
                                                        const RuntimeOptions& options,
                                                        const std::string& model_sha256) {
  v1::SceneControlGraphEnvelope envelope;
  envelope.set_schema_major(1U);
  envelope.set_schema_minor(0U);
  auto* producer = envelope.mutable_producer();
  producer->set_git_commit(options.producer.git_commit);
  producer->set_git_dirty(options.producer.git_dirty);
  producer->set_model_artifact_sha256(model_sha256);
  producer->set_configuration_sha256(options.producer.configuration_sha256);
  producer->set_runtime_build_sha256(options.producer.runtime_build_sha256);
  producer->set_execution_provider_profile("cpu-fp32");
  producer->set_provider_assignment_digest(Sha256Text(kCpuProviderDigestInput));
  auto* graph = envelope.mutable_graph();
  graph->set_role(v1::GRAPH_ROLE_PREDICTION);
  graph->mutable_frame_key()->CopyFrom(inputs.output_sensor_frame.frame_key());
  graph->mutable_sensor_frame()->CopyFrom(inputs.output_sensor_frame);
  const PostprocessMaps maps = AddNodes(outputs, options, *graph);
  AddEdges(outputs, options, maps, *graph);
  const auto validation = contract::Validate(envelope);
  if (!validation.valid) {
    throw RuntimeError("RUNTIME_OUTPUT_CONTRACT", validation.reason_code + " at " +
                                                      validation.path + ": " + validation.detail);
  }
  return envelope;
}

}  // namespace

class CpuRuntime::Impl final {
 public:
  explicit Impl(RuntimeOptions selected_options)
      : options(std::move(selected_options)),
        environment(ORT_LOGGING_LEVEL_WARNING, "junctionlens-runtime"),
        session_options(),
        model_sha256(Sha256File(options.model_path)) {
    ValidateOptions(options);
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    session_options.SetIntraOpNumThreads(1);
    session_options.SetInterOpNumThreads(1);
    session_options.AppendExecutionProvider_CPU(1);
    session =
        std::make_unique<Ort::Session>(environment, options.model_path.c_str(), session_options);
    ValidateTensorContract(*session, true);
    ValidateTensorContract(*session, false);
    ValidateMetadata(ReadMetadata(*session), options);
  }

  RuntimeOptions options;
  Ort::Env environment;
  Ort::SessionOptions session_options;
  std::string model_sha256;
  std::unique_ptr<Ort::Session> session;
};

CpuRuntime::CpuRuntime(RuntimeOptions options)
    : impl_(std::make_unique<Impl>(std::move(options))) {}

CpuRuntime::CpuRuntime(CpuRuntime&&) noexcept = default;

CpuRuntime& CpuRuntime::operator=(CpuRuntime&&) noexcept = default;

CpuRuntime::~CpuRuntime() = default;

v1::SceneControlGraphEnvelope CpuRuntime::Infer(const PreprocessedInputs& inputs,
                                                BufferLease& lease) const {
  constexpr std::size_t kExpectedImageValues = 2U * 8U * 3U * 384U * 640U;
  if (inputs.images.size() != kExpectedImageValues || inputs.camera_valid.size() != 16U ||
      inputs.intrinsics.size() != 144U || inputs.t_vehicle_camera.size() != 256U ||
      inputs.ego_motion_previous_to_current.size() != 16U || inputs.temporal_valid.size() != 1U) {
    throw RuntimeError("RUNTIME_INPUT_SHAPE",
                       "preprocessed buffers differ from the frozen profile");
  }
  Ort::MemoryInfo memory = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  const std::vector<std::int64_t> image_shape = {1, 2, 8, 3, 384, 640};
  const std::vector<std::int64_t> valid_shape = {1, 2, 8};
  const std::vector<std::int64_t> intrinsic_shape = {1, 2, 8, 3, 3};
  const std::vector<std::int64_t> transform_shape = {1, 2, 8, 4, 4};
  const std::vector<std::int64_t> ego_shape = {1, 4, 4};
  const std::vector<std::int64_t> temporal_shape = {1};
  std::vector<Ort::Value> input_values;
  input_values.reserve(InputSpecs().size());
  input_values.emplace_back(MakeTensor(memory, inputs.images, image_shape));
  input_values.emplace_back(MakeBoolTensor(memory, inputs.camera_valid, valid_shape));
  input_values.emplace_back(MakeTensor(memory, inputs.intrinsics, intrinsic_shape));
  input_values.emplace_back(MakeTensor(memory, inputs.t_vehicle_camera, transform_shape));
  input_values.emplace_back(MakeTensor(memory, inputs.ego_motion_previous_to_current, ego_shape));
  input_values.emplace_back(MakeBoolTensor(memory, inputs.temporal_valid, temporal_shape));
  std::vector<const char*> input_names;
  std::vector<const char*> output_names;
  for (const TensorSpec& spec : InputSpecs()) {
    input_names.push_back(spec.name.data());
  }
  for (const TensorSpec& spec : OutputSpecs()) {
    output_names.push_back(spec.name.data());
  }
  const std::vector<Ort::Value> outputs =
      impl_->session->Run(Ort::RunOptions{nullptr}, input_names.data(), input_values.data(),
                          input_values.size(), output_names.data(), output_names.size());
  if (outputs.size() != OutputSpecs().size()) {
    throw RuntimeError("RUNTIME_OUTPUT_COUNT", "runtime output count differs from the contract");
  }
  for (std::size_t index = 0; index < outputs.size(); ++index) {
    static_cast<void>(View(outputs, index));
  }
  lease.Advance(BufferState::kPostprocessing);
  return Postprocess(outputs, inputs, impl_->options, impl_->model_sha256);
}

const RuntimeOptions& CpuRuntime::options() const noexcept { return impl_->options; }

}  // namespace junctionlens::infer
