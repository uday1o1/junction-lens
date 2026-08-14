#include <filesystem>
#include <iostream>
#include <string>

#include <google/protobuf/util/json_util.h>

#include "contract/validation.hpp"

namespace {

[[nodiscard]] std::string EscapeJson(const std::string& input) {
  std::string result;
  for (const char character : input) {
    switch (character) {
      case '\\':
        result += "\\\\";
        break;
      case '"':
        result += "\\\"";
        break;
      case '\n':
        result += "\\n";
        break;
      default:
        result += character;
        break;
    }
  }
  return result;
}

void Usage() {
  std::cout << "Usage: junctionlens-contract-probe --input FILE [--emit-json]\n";
}

}  // namespace

int main(int argc, char** argv) {
  junctionlens::contract::VerifyExactProtobufRuntime();
  std::filesystem::path input;
  bool emit_json = false;
  for (int index = 1; index < argc; ++index) {
    const std::string argument = argv[index];
    if (argument == "--help") {
      Usage();
      return 0;
    }
    if (argument == "--emit-json") {
      emit_json = true;
    } else if (argument == "--input" && index + 1 < argc) {
      input = argv[++index];
    } else {
      Usage();
      return 2;
    }
  }
  if (input.empty()) {
    Usage();
    return 2;
  }
  junctionlens::v1::SceneControlGraphEnvelope envelope;
  const auto result = junctionlens::contract::ParseFile(input, envelope);
  if (!result.valid) {
    std::cerr << "{\"detail\":\"" << EscapeJson(result.detail) << "\",\"path\":\""
              << EscapeJson(result.path) << "\",\"reason_code\":\""
              << EscapeJson(result.reason_code) << "\"}\n";
    return 2;
  }
  if (emit_json) {
    google::protobuf::util::JsonPrintOptions options;
    options.preserve_proto_field_names = true;
    options.always_print_fields_with_no_presence = true;
    std::string output;
    const auto status = google::protobuf::util::MessageToJsonString(envelope, &output, options);
    if (!status.ok()) {
      std::cerr << "failed to render ProtoJSON: " << status << '\n';
      return 2;
    }
    std::cout << output << '\n';
    return 0;
  }
  const auto& graph = envelope.graph();
  std::cout << "{\"edges\":" << graph.edges_size() << ",\"lanes\":" << graph.lanes_size()
            << ",\"road_areas\":" << graph.road_areas_size() << ",\"schema_major\":"
            << envelope.schema_major() << ",\"schema_minor\":" << envelope.schema_minor()
            << ",\"traffic_controls\":" << graph.traffic_controls_size() << "}\n";
  return 0;
}
