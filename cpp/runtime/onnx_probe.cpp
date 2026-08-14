#include <cpu_provider_factory.h>
#include <onnxruntime_cxx_api.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

constexpr std::string_view kInputContractSha256 =
    "cf5adc1545fa9b82f2a4429adab5c020bc64bd0357c908e5457f88dd62ea34ef";
constexpr std::string_view kOutputContractSha256 =
    "84081fbd524a0439d2ec5f6c26fb33baf72cea11dceefd85241f7f4750b5f495";

struct TensorSpec {
  std::string_view name;
  ONNXTensorElementDataType element_type;
  std::vector<std::int64_t> dimensions;
};

struct Arguments {
  std::string model_path;
  std::string expected_profile_sha256;
  std::int64_t frame_index = 7;
};

[[nodiscard]] std::string JsonEscape(const std::string_view value) {
  std::ostringstream output;
  for (const char character : value) {
    switch (character) {
      case '"':
        output << "\\\"";
        break;
      case '\\':
        output << "\\\\";
        break;
      case '\n':
        output << "\\n";
        break;
      case '\r':
        output << "\\r";
        break;
      case '\t':
        output << "\\t";
        break;
      default:
        if (static_cast<unsigned char>(character) < 0x20U) {
          output << "\\u" << std::hex << std::setw(4) << std::setfill('0')
                 << static_cast<unsigned int>(static_cast<unsigned char>(character)) << std::dec;
        } else {
          output << character;
        }
    }
  }
  return output.str();
}

[[nodiscard]] Arguments ParseArguments(const int argc, char** argv) {
  Arguments result;
  for (int index = 1; index < argc; ++index) {
    const std::string_view argument(argv[index]);
    if ((argument == "--model" || argument == "--expected-profile-sha256" ||
         argument == "--frame-index") &&
        index + 1 >= argc) {
      throw std::invalid_argument("missing value after " + std::string(argument));
    }
    if (argument == "--model") {
      result.model_path = argv[++index];
    } else if (argument == "--expected-profile-sha256") {
      result.expected_profile_sha256 = argv[++index];
    } else if (argument == "--frame-index") {
      const std::string value(argv[++index]);
      std::size_t consumed = 0;
      result.frame_index = std::stoll(value, &consumed);
      if (consumed != value.size()) {
        throw std::invalid_argument("frame index must be an integer");
      }
    } else if (argument == "--help") {
      std::cout << "Usage: junctionlens-onnx-probe --model PATH "
                   "--expected-profile-sha256 HEX [--frame-index 0..31]\n";
      std::exit(EXIT_SUCCESS);
    } else {
      throw std::invalid_argument("unknown argument: " + std::string(argument));
    }
  }
  if (result.model_path.empty() || result.expected_profile_sha256.empty()) {
    throw std::invalid_argument("--model and --expected-profile-sha256 are required");
  }
  if (result.frame_index < 0 || result.frame_index > 31) {
    throw std::invalid_argument("frame index must be between 0 and 31");
  }
  return result;
}

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

void ValidateTensorContract(const Ort::Session& session, const bool input) {
  Ort::AllocatorWithDefaultOptions allocator;
  const auto& specs = input ? InputSpecs() : OutputSpecs();
  const std::size_t observed_count = input ? session.GetInputCount() : session.GetOutputCount();
  if (observed_count != specs.size()) {
    throw std::runtime_error("tensor count differs from the frozen contract");
  }
  for (std::size_t index = 0; index < specs.size(); ++index) {
    auto name = input ? session.GetInputNameAllocated(index, allocator)
                      : session.GetOutputNameAllocated(index, allocator);
    if (name.get() == nullptr || std::string_view(name.get()) != specs[index].name) {
      throw std::runtime_error("tensor name differs from the frozen contract at index " +
                               std::to_string(index));
    }
    const Ort::TypeInfo type_info =
        input ? session.GetInputTypeInfo(index) : session.GetOutputTypeInfo(index);
    const auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
    if (tensor_info.GetElementType() != specs[index].element_type) {
      throw std::runtime_error("tensor element type differs for " + std::string(specs[index].name));
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
      throw std::runtime_error("tensor shape differs for " + std::string(specs[index].name));
    }
  }
}

[[nodiscard]] std::map<std::string, std::string> ReadMetadata(const Ort::Session& session) {
  Ort::AllocatorWithDefaultOptions allocator;
  const Ort::ModelMetadata metadata = session.GetModelMetadata();
  std::map<std::string, std::string> result;
  for (auto& key : metadata.GetCustomMetadataMapKeysAllocated(allocator)) {
    if (key.get() == nullptr) {
      continue;
    }
    auto value = metadata.LookupCustomMetadataMapAllocated(key.get(), allocator);
    if (value.get() != nullptr) {
      result.emplace(key.get(), value.get());
    }
  }
  return result;
}

void RequireMetadata(const std::map<std::string, std::string>& metadata,
                     const std::string_view key,
                     const std::string_view expected) {
  const auto found = metadata.find(std::string(key));
  if (found == metadata.end()) {
    throw std::runtime_error("required model metadata is absent: " + std::string(key));
  }
  if (found->second != expected) {
    throw std::runtime_error("model metadata differs: " + std::string(key));
  }
}

void ValidateMetadata(const std::map<std::string, std::string>& metadata,
                      const Arguments& arguments) {
  RequireMetadata(metadata, "junctionlens.schema_version", "1.0.0");
  RequireMetadata(metadata, "junctionlens.profile_id", "m0-feasibility-spike-v1");
  RequireMetadata(metadata, "junctionlens.profile_sha256", arguments.expected_profile_sha256);
  RequireMetadata(metadata, "junctionlens.input_contract_sha256", kInputContractSha256);
  RequireMetadata(metadata, "junctionlens.output_contract_sha256", kOutputContractSha256);
  RequireMetadata(metadata, "junctionlens.opset", "18");
  RequireMetadata(metadata, "junctionlens.precision", "fp32");
  const auto checkpoint = metadata.find("junctionlens.checkpoint_sha256");
  if (checkpoint == metadata.end() || checkpoint->second.size() != 64U) {
    throw std::runtime_error("checkpoint metadata is absent or malformed");
  }
}

[[nodiscard]] std::size_t FlatIndex(const std::vector<std::int64_t>& shape,
                                    const std::vector<std::int64_t>& indices) {
  if (shape.size() != indices.size()) {
    throw std::logic_error("rank mismatch in synthetic input generation");
  }
  std::size_t offset = 0;
  for (std::size_t dimension = 0; dimension < shape.size(); ++dimension) {
    offset *= static_cast<std::size_t>(shape[dimension]);
    offset += static_cast<std::size_t>(indices[dimension]);
  }
  return offset;
}

[[nodiscard]] std::size_t ElementCount(const std::vector<std::int64_t>& shape) {
  std::size_t result = 1;
  for (const std::int64_t dimension : shape) {
    if (dimension <= 0) {
      throw std::logic_error("concrete tensor dimensions must be positive");
    }
    result *= static_cast<std::size_t>(dimension);
  }
  return result;
}

struct Inputs {
  std::vector<float> images;
  std::vector<std::uint8_t> camera_valid;
  std::vector<float> intrinsics;
  std::vector<float> transforms;
  std::vector<float> ego_motion;
  std::vector<std::uint8_t> temporal_valid;
};

[[nodiscard]] Inputs MakeInputs(const std::int64_t frame_index) {
  const std::vector<std::int64_t> image_shape = {1, 2, 8, 3, 384, 640};
  const std::vector<std::int64_t> valid_shape = {1, 2, 8};
  const std::vector<std::int64_t> intrinsic_shape = {1, 2, 8, 3, 3};
  const std::vector<std::int64_t> transform_shape = {1, 2, 8, 4, 4};
  Inputs result{
      std::vector<float>(ElementCount(image_shape), 0.0F),
      std::vector<std::uint8_t>(ElementCount(valid_shape), 1U),
      std::vector<float>(ElementCount(intrinsic_shape), 0.0F),
      std::vector<float>(ElementCount(transform_shape), 0.0F),
      std::vector<float>(16U, 0.0F),
      std::vector<std::uint8_t>(1U, 1U),
  };
  for (std::int64_t camera = 0; camera < 8; ++camera) {
    const bool bit = ((frame_index >> camera) & 1) != 0;
    const float bit_value = bit ? 1.0F : 0.0F;
    for (std::int64_t row = 0; row < 384; ++row) {
      for (std::int64_t column = 0; column < 640; ++column) {
        result.images[FlatIndex(image_shape, {0, 1, camera, 0, row, column})] = bit_value;
        result.images[FlatIndex(image_shape, {0, 1, camera, 1, row, column})] = 1.0F - bit_value;
        result.images[FlatIndex(image_shape, {0, 0, camera, 2, row, column})] =
            bit_value * 0.5F;
      }
    }
    for (std::int64_t timestamp = 0; timestamp < 2; ++timestamp) {
      result.intrinsics[FlatIndex(intrinsic_shape, {0, timestamp, camera, 0, 0})] = 640.0F;
      result.intrinsics[FlatIndex(intrinsic_shape, {0, timestamp, camera, 1, 1})] = 384.0F;
      result.intrinsics[FlatIndex(intrinsic_shape, {0, timestamp, camera, 0, 2})] = 320.0F;
      result.intrinsics[FlatIndex(intrinsic_shape, {0, timestamp, camera, 1, 2})] = 192.0F;
      result.intrinsics[FlatIndex(intrinsic_shape, {0, timestamp, camera, 2, 2})] = 1.0F;
      for (std::int64_t axis = 0; axis < 4; ++axis) {
        result.transforms[FlatIndex(transform_shape, {0, timestamp, camera, axis, axis})] = 1.0F;
      }
      result.transforms[FlatIndex(transform_shape, {0, timestamp, camera, 1, 3})] =
          -1.75F + static_cast<float>(camera) * 0.5F;
    }
  }
  for (std::size_t axis = 0; axis < 4U; ++axis) {
    result.ego_motion[axis * 4U + axis] = 1.0F;
  }
  result.ego_motion[3U] = static_cast<float>(frame_index) / 100.0F;
  return result;
}

template <typename Element>
[[nodiscard]] Ort::Value MakeTensor(Ort::MemoryInfo& memory,
                                    std::vector<Element>& values,
                                    const std::vector<std::int64_t>& shape) {
  return Ort::Value::CreateTensor<Element>(
      memory, values.data(), values.size(), shape.data(), shape.size());
}

[[nodiscard]] Ort::Value MakeBoolTensor(Ort::MemoryInfo& memory,
                                        std::vector<std::uint8_t>& values,
                                        const std::vector<std::int64_t>& shape) {
  return Ort::Value::CreateTensor(memory,
                                  values.data(),
                                  values.size() * sizeof(std::uint8_t),
                                  shape.data(),
                                  shape.size(),
                                  ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL);
}

void WriteStringArray(const std::vector<std::string>& values) {
  std::cout << '[';
  for (std::size_t index = 0; index < values.size(); ++index) {
    if (index != 0U) {
      std::cout << ',';
    }
    std::cout << '"' << JsonEscape(values[index]) << '"';
  }
  std::cout << ']';
}

void WriteShape(const std::vector<std::int64_t>& shape) {
  std::cout << '[';
  for (std::size_t index = 0; index < shape.size(); ++index) {
    if (index != 0U) {
      std::cout << ',';
    }
    std::cout << shape[index];
  }
  std::cout << ']';
}

void WriteOutputs(const std::vector<Ort::Value>& outputs) {
  std::cout << "\"outputs\":[";
  for (std::size_t index = 0; index < outputs.size(); ++index) {
    if (index != 0U) {
      std::cout << ',';
    }
    const auto info = outputs[index].GetTensorTypeAndShapeInfo();
    if (info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
      throw std::runtime_error("native output was not float32");
    }
    const std::size_t count = info.GetElementCount();
    const float* data = outputs[index].GetTensorData<float>();
    std::cout << "{\"name\":\"" << JsonEscape(std::string(OutputSpecs()[index].name))
              << "\",\"shape\":";
    WriteShape(info.GetShape());
    std::cout << ",\"values\":[";
    for (std::size_t element = 0; element < count; ++element) {
      if (!std::isfinite(data[element])) {
        throw std::runtime_error("native output contains a nonfinite value");
      }
      if (element != 0U) {
        std::cout << ',';
      }
      std::cout << std::setprecision(std::numeric_limits<float>::max_digits10) << data[element];
    }
    std::cout << "]}";
  }
  std::cout << ']';
}

int Run(const Arguments& arguments) {
  Ort::Env environment(ORT_LOGGING_LEVEL_WARNING, "junctionlens-onnx-probe");
  Ort::SessionOptions options;
  options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
  options.SetIntraOpNumThreads(1);
  options.SetInterOpNumThreads(1);
  options.AppendExecutionProvider_CPU(1);
  Ort::Session session(environment, arguments.model_path.c_str(), options);
  ValidateTensorContract(session, true);
  ValidateTensorContract(session, false);
  const auto metadata = ReadMetadata(session);
  ValidateMetadata(metadata, arguments);

  Inputs input_storage = MakeInputs(arguments.frame_index);
  Ort::MemoryInfo memory = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  const std::vector<std::int64_t> image_shape = {1, 2, 8, 3, 384, 640};
  const std::vector<std::int64_t> valid_shape = {1, 2, 8};
  const std::vector<std::int64_t> intrinsic_shape = {1, 2, 8, 3, 3};
  const std::vector<std::int64_t> transform_shape = {1, 2, 8, 4, 4};
  const std::vector<std::int64_t> ego_shape = {1, 4, 4};
  const std::vector<std::int64_t> temporal_shape = {1};
  std::vector<Ort::Value> input_values;
  input_values.reserve(InputSpecs().size());
  input_values.emplace_back(MakeTensor(memory, input_storage.images, image_shape));
  input_values.emplace_back(MakeBoolTensor(memory, input_storage.camera_valid, valid_shape));
  input_values.emplace_back(MakeTensor(memory, input_storage.intrinsics, intrinsic_shape));
  input_values.emplace_back(MakeTensor(memory, input_storage.transforms, transform_shape));
  input_values.emplace_back(MakeTensor(memory, input_storage.ego_motion, ego_shape));
  input_values.emplace_back(MakeBoolTensor(memory, input_storage.temporal_valid, temporal_shape));

  std::vector<const char*> input_names;
  std::vector<const char*> output_names;
  input_names.reserve(InputSpecs().size());
  output_names.reserve(OutputSpecs().size());
  for (const TensorSpec& spec : InputSpecs()) {
    input_names.push_back(spec.name.data());
  }
  for (const TensorSpec& spec : OutputSpecs()) {
    output_names.push_back(spec.name.data());
  }
  const std::vector<Ort::Value> outputs = session.Run(Ort::RunOptions{nullptr},
                                                       input_names.data(),
                                                       input_values.data(),
                                                       input_values.size(),
                                                       output_names.data(),
                                                       output_names.size());
  if (outputs.size() != OutputSpecs().size()) {
    throw std::runtime_error("runtime output count differs from the frozen contract");
  }

  std::cout << "{\"schema_version\":\"1.0.0\",\"status\":\"PASSED\",";
  std::cout << "\"runtime_version\":\"" << JsonEscape(Ort::GetVersionString()) << "\",";
  std::cout << "\"requested_provider\":\"CPUExecutionProvider\",\"available_providers\":";
  WriteStringArray(Ort::GetAvailableProviders());
  std::cout << ",\"profile_sha256\":\"" << JsonEscape(arguments.expected_profile_sha256)
            << "\",";
  WriteOutputs(outputs);
  std::cout << "}\n";
  return EXIT_SUCCESS;
}

}  // namespace

int main(const int argc, char** argv) {
  try {
    return Run(ParseArguments(argc, argv));
  } catch (const Ort::Exception& error) {
    std::cerr << "onnx runtime error: " << error.what() << '\n';
  } catch (const std::exception& error) {
    std::cerr << "onnx probe error: " << error.what() << '\n';
  }
  return EXIT_FAILURE;
}
