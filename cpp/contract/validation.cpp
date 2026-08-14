#include "contract/validation.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iterator>
#include <limits>
#include <set>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <google/protobuf/descriptor.h>
#include <google/protobuf/message.h>
#include <google/protobuf/stubs/common.h>

static_assert(GOOGLE_PROTOBUF_VERSION == 6031001, "JunctionLens requires protobuf 6.31.1");

namespace junctionlens::contract {
namespace {

using Envelope = v1::SceneControlGraphEnvelope;

[[nodiscard]] ValidationResult Pass() { return {true, "OK", "", ""}; }

[[nodiscard]] ValidationResult Fail(
    std::string reason_code,
    std::string path,
    std::string detail
) {
  return {false, std::move(reason_code), std::move(path), std::move(detail)};
}

[[nodiscard]] ValidationResult ValidateFinite(
    const google::protobuf::Message& message,
    const std::string& path
) {
  const auto* reflection = message.GetReflection();
  std::vector<const google::protobuf::FieldDescriptor*> fields;
  reflection->ListFields(message, &fields);
  for (const auto* field : fields) {
    const int count = field->is_repeated() ? reflection->FieldSize(message, field) : 1;
    for (int index = 0; index < count; ++index) {
      const std::string field_name(field->name().data(), field->name().size());
      const std::string field_path = path + "." + field_name
          + (field->is_repeated() ? "[" + std::to_string(index) + "]" : "");
      if (field->cpp_type() == google::protobuf::FieldDescriptor::CPPTYPE_MESSAGE) {
        const auto& child = field->is_repeated()
            ? reflection->GetRepeatedMessage(message, field, index)
            : reflection->GetMessage(message, field);
        auto result = ValidateFinite(child, field_path);
        if (!result.valid) {
          return result;
        }
      } else if (field->cpp_type() == google::protobuf::FieldDescriptor::CPPTYPE_DOUBLE) {
        const double value = field->is_repeated()
            ? reflection->GetRepeatedDouble(message, field, index)
            : reflection->GetDouble(message, field);
        if (!std::isfinite(value)) {
          return Fail("CONTRACT_NONFINITE", field_path, "floating-point values must be finite");
        }
      } else if (field->cpp_type() == google::protobuf::FieldDescriptor::CPPTYPE_FLOAT) {
        const float value = field->is_repeated()
            ? reflection->GetRepeatedFloat(message, field, index)
            : reflection->GetFloat(message, field);
        if (!std::isfinite(value)) {
          return Fail("CONTRACT_NONFINITE", field_path, "floating-point values must be finite");
        }
      }
    }
  }
  return Pass();
}

[[nodiscard]] ValidationResult ValidateTransform(
    const v1::Matrix4d& transform,
    const std::string& path
) {
  if (transform.values_size() != 16) {
    return Fail("CONTRACT_TRANSFORM_SHAPE", path, "matrix requires exactly 16 values");
  }
  constexpr std::array<double, 4> expected{0.0, 0.0, 0.0, 1.0};
  for (std::size_t index = 0; index < expected.size(); ++index) {
    if (std::abs(transform.values(static_cast<int>(12U + index)) - expected[index]) > 1.0e-8) {
      return Fail("CONTRACT_TRANSFORM_AFFINE", path, "last row must be [0, 0, 0, 1]");
    }
  }
  for (int row = 0; row < 3; ++row) {
    for (int column = 0; column < 3; ++column) {
      double dot = 0.0;
      for (int coordinate = 0; coordinate < 3; ++coordinate) {
        dot += transform.values(row * 4 + coordinate)
            * transform.values(column * 4 + coordinate);
      }
      const double target = row == column ? 1.0 : 0.0;
      if (std::abs(dot - target) > 1.0e-6) {
        return Fail("CONTRACT_TRANSFORM_RIGID", path, "rotation must be orthonormal");
      }
    }
  }
  return Pass();
}

[[nodiscard]] int EncodedNodeType(std::uint64_t node_id) {
  return static_cast<int>(node_id >> 56U);
}

[[nodiscard]] ValidationResult ValidateNodes(
    const v1::SceneControlGraph& graph,
    std::unordered_map<std::uint64_t, v1::NodeType>& node_types
) {
  const auto add_node = [&](std::uint64_t node_id, v1::NodeType node_type, const std::string& path) {
    if (node_id == 0U) {
      return Fail("CONTRACT_NODE_ID_ZERO", path, "node ID must be nonzero");
    }
    if (!node_types.emplace(node_id, node_type).second) {
      return Fail("CONTRACT_NODE_ID_DUPLICATE", path, "node IDs are frame-unique");
    }
    if (graph.role() == v1::GRAPH_ROLE_PREDICTION && EncodedNodeType(node_id) != node_type) {
      return Fail("CONTRACT_NODE_ID_TYPE", path, "encoded type does not match node type");
    }
    return Pass();
  };
  for (int index = 0; index < graph.lanes_size(); ++index) {
    auto result = add_node(
        graph.lanes(index).node_id(),
        v1::NODE_TYPE_LANE_SEGMENT,
        "graph.lanes[" + std::to_string(index) + "].node_id"
    );
    if (!result.valid) {
      return result;
    }
  }
  for (int index = 0; index < graph.traffic_controls_size(); ++index) {
    const auto& control = graph.traffic_controls(index);
    const std::string path = "graph.traffic_controls[" + std::to_string(index) + "]";
    auto result = add_node(control.node_id(), v1::NODE_TYPE_TRAFFIC_CONTROL, path + ".node_id");
    if (!result.valid) {
      return result;
    }
    const auto& box = control.normalized_half_open_box();
    if (!(box.x_min() >= 0.0 && box.x_min() < box.x_max() && box.x_max() <= 1.0
          && box.y_min() >= 0.0 && box.y_min() < box.y_max() && box.y_max() <= 1.0)) {
      return Fail(
          "CONTRACT_NORMALIZED_BOX",
          path + ".normalized_half_open_box",
          "half-open box must have positive area inside [0, 1]"
      );
    }
  }
  for (int index = 0; index < graph.road_areas_size(); ++index) {
    auto result = add_node(
        graph.road_areas(index).node_id(),
        v1::NODE_TYPE_ROAD_AREA,
        "graph.road_areas[" + std::to_string(index) + "].node_id"
    );
    if (!result.valid) {
      return result;
    }
  }
  return Pass();
}

[[nodiscard]] ValidationResult ValidateEdges(
    const v1::SceneControlGraph& graph,
    const std::unordered_map<std::uint64_t, v1::NodeType>& node_types
) {
  std::set<std::uint64_t> edge_ids;
  for (int index = 0; index < graph.edges_size(); ++index) {
    const auto& edge = graph.edges(index);
    const std::string path = "graph.edges[" + std::to_string(index) + "]";
    if (edge.edge_id() == 0U) {
      return Fail("CONTRACT_EDGE_ID_ZERO", path + ".edge_id", "edge ID must be nonzero");
    }
    if (!edge_ids.insert(edge.edge_id()).second) {
      return Fail("CONTRACT_EDGE_ID_DUPLICATE", path + ".edge_id", "edge IDs must be unique");
    }
    const auto source = node_types.find(edge.source_node_id());
    const auto target = node_types.find(edge.target_node_id());
    if (source == node_types.end() || target == node_types.end()) {
      return Fail("CONTRACT_EDGE_DANGLING", path, "edge endpoint does not exist");
    }
    const bool lane_successor = edge.edge_type() == v1::GRAPH_EDGE_TYPE_LANE_SUCCESSOR
        && source->second == v1::NODE_TYPE_LANE_SEGMENT
        && target->second == v1::NODE_TYPE_LANE_SEGMENT;
    const bool control_lane = edge.edge_type() == v1::GRAPH_EDGE_TYPE_CONTROL_APPLIES_TO_LANE
        && source->second == v1::NODE_TYPE_TRAFFIC_CONTROL
        && target->second == v1::NODE_TYPE_LANE_SEGMENT;
    if (!lane_successor && !control_lane) {
      return Fail("CONTRACT_EDGE_TYPES", path, "edge endpoint types do not match edge type");
    }
  }
  return Pass();
}

}  // namespace

void VerifyExactProtobufRuntime() { GOOGLE_PROTOBUF_VERIFY_VERSION; }

ValidationResult Validate(const Envelope& envelope) {
  auto result = ValidateFinite(envelope, "envelope");
  if (!result.valid) {
    return result;
  }
  if (envelope.schema_major() != 1U) {
    return Fail("CONTRACT_SCHEMA_MAJOR", "schema_major", "expected 1");
  }
  const auto& graph = envelope.graph();
  if (graph.role() == v1::GRAPH_ROLE_UNSPECIFIED) {
    return Fail("CONTRACT_ENUM_UNSPECIFIED", "graph.role", "graph role is required");
  }
  if (graph.has_sensor_frame()) {
    result = ValidateTransform(graph.sensor_frame().t_world_vehicle(), "graph.sensor_frame.t_world_vehicle");
    if (!result.valid) {
      return result;
    }
    for (int index = 0; index < graph.sensor_frame().cameras_size(); ++index) {
      result = ValidateTransform(
          graph.sensor_frame().cameras(index).t_vehicle_camera(),
          "graph.sensor_frame.cameras[" + std::to_string(index) + "].t_vehicle_camera"
      );
      if (!result.valid) {
        return result;
      }
    }
  }
  std::unordered_map<std::uint64_t, v1::NodeType> node_types;
  result = ValidateNodes(graph, node_types);
  if (!result.valid) {
    return result;
  }
  return ValidateEdges(graph, node_types);
}

ValidationResult ParseFile(const std::filesystem::path& path, Envelope& envelope) {
  std::ifstream input(path, std::ios::binary);
  if (!input) {
    return Fail("CONTRACT_IO", "binary", "input could not be opened");
  }
  input.seekg(0, std::ios::end);
  const auto end = input.tellg();
  if (end < 0) {
    return Fail("CONTRACT_IO", "binary", "input size could not be determined");
  }
  const auto size = static_cast<std::uint64_t>(end);
  if (size > kMaximumSerializedBytes || size > static_cast<std::uint64_t>(std::numeric_limits<int>::max())) {
    return Fail("CONTRACT_SIZE_LIMIT", "binary", "payload exceeds 67108864 bytes");
  }
  input.seekg(0, std::ios::beg);
  std::string payload{std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
  if (!envelope.ParseFromArray(payload.data(), static_cast<int>(payload.size()))) {
    return Fail("CONTRACT_BINARY_MALFORMED", "binary", "protobuf parse failed");
  }
  return Validate(envelope);
}

}  // namespace junctionlens::contract
