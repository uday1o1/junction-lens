cmake_minimum_required(VERSION 3.31)

project(JunctionLens VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

include(cmake/dependencies.cmake)
include(CTest)

add_executable(junctionlens-toolchain-probe cpp/tools/toolchain_probe.cpp)

set(JUNCTIONLENS_PROTO_ROOT "${PROJECT_SOURCE_DIR}/proto")
set(JUNCTIONLENS_PROTO_FILE "${JUNCTIONLENS_PROTO_ROOT}/junctionlens/v1/scene_control_graph.proto")
set(JUNCTIONLENS_GENERATED_ROOT "${PROJECT_BINARY_DIR}/generated")
set(
  JUNCTIONLENS_GENERATED_PROTO_SOURCE
  "${JUNCTIONLENS_GENERATED_ROOT}/junctionlens/v1/scene_control_graph.pb.cc"
)
set(
  JUNCTIONLENS_GENERATED_PROTO_HEADER
  "${JUNCTIONLENS_GENERATED_ROOT}/junctionlens/v1/scene_control_graph.pb.h"
)

add_custom_command(
  OUTPUT
    "${JUNCTIONLENS_GENERATED_PROTO_SOURCE}"
    "${JUNCTIONLENS_GENERATED_PROTO_HEADER}"
  COMMAND "${PROJECT_SOURCE_DIR}/.tools/bin/protoc"
    "--proto_path=${JUNCTIONLENS_PROTO_ROOT}"
    "--cpp_out=${JUNCTIONLENS_GENERATED_ROOT}"
    "${JUNCTIONLENS_PROTO_FILE}"
  DEPENDS "${JUNCTIONLENS_PROTO_FILE}"
  COMMENT "Generating exact JunctionLens V1 C++ protobuf bindings"
  VERBATIM
)

add_library(
  junctionlens_contract_proto
  STATIC
  "${JUNCTIONLENS_GENERATED_PROTO_SOURCE}"
)
target_include_directories(
  junctionlens_contract_proto
  PUBLIC
  "${JUNCTIONLENS_GENERATED_ROOT}"
)
target_link_libraries(junctionlens_contract_proto PUBLIC protobuf::libprotobuf)

add_library(junctionlens_contract STATIC cpp/contract/validation.cpp)
add_library(junctionlens::contract ALIAS junctionlens_contract)
target_include_directories(junctionlens_contract PUBLIC "${PROJECT_SOURCE_DIR}/cpp")
target_link_libraries(junctionlens_contract PUBLIC junctionlens_contract_proto)

add_executable(junctionlens-contract-probe cpp/contract/probe.cpp)
target_link_libraries(junctionlens-contract-probe PRIVATE junctionlens::contract)

if(NOT TARGET onnxruntime::onnxruntime)
  message(
    FATAL_ERROR
    "The locked ONNX Runtime C++ archive is absent. Run ./tools/jl bootstrap-cpu."
  )
endif()

add_executable(junctionlens-onnx-probe cpp/runtime/onnx_probe.cpp)
target_link_libraries(junctionlens-onnx-probe PRIVATE onnxruntime::onnxruntime)
set_target_properties(
  junctionlens-onnx-probe
  PROPERTIES
    BUILD_RPATH "${JUNCTIONLENS_ONNXRUNTIME_ROOT}/lib"
    INSTALL_RPATH "$ORIGIN/../lib"
)

if(MSVC)
  target_compile_options(junctionlens-toolchain-probe PRIVATE /W4 /WX)
  target_compile_options(junctionlens-onnx-probe PRIVATE /W4 /WX)
  target_compile_options(junctionlens_contract PRIVATE /W4 /WX)
  target_compile_options(junctionlens-contract-probe PRIVATE /W4 /WX)
else()
  set(
    JUNCTIONLENS_WARNING_OPTIONS
      -Wall
      -Wextra
      -Wpedantic
      -Wconversion
      -Wshadow
      -Werror
  )
  target_compile_options(junctionlens-toolchain-probe PRIVATE ${JUNCTIONLENS_WARNING_OPTIONS})
  target_compile_options(junctionlens-onnx-probe PRIVATE ${JUNCTIONLENS_WARNING_OPTIONS})
  target_compile_options(junctionlens_contract PRIVATE ${JUNCTIONLENS_WARNING_OPTIONS})
  target_compile_options(junctionlens-contract-probe PRIVATE ${JUNCTIONLENS_WARNING_OPTIONS})
  if(APPLE)
    set(JUNCTIONLENS_APPLE_THIRD_PARTY_WARNINGS -Wno-gcc-compat -Wno-nullability-extension)
    target_compile_options(junctionlens_contract PRIVATE ${JUNCTIONLENS_APPLE_THIRD_PARTY_WARNINGS})
    target_compile_options(junctionlens-contract-probe PRIVATE ${JUNCTIONLENS_APPLE_THIRD_PARTY_WARNINGS})
  endif()
endif()

if(BUILD_TESTING)
  add_test(NAME toolchain-probe-json COMMAND junctionlens-toolchain-probe)
  add_test(NAME onnx-probe-help COMMAND junctionlens-onnx-probe --help)
  add_test(NAME contract-probe-help COMMAND junctionlens-contract-probe --help)
  add_test(
    NAME contract-golden-cpp
    COMMAND junctionlens-contract-probe
      --input "${PROJECT_SOURCE_DIR}/tests/fixtures/contract/v1/golden.pb"
  )
  set_tests_properties(
    toolchain-probe-json
    PROPERTIES
      PASS_REGULAR_EXPRESSION
      "\"cxx_standard\":202002"
  )
endif()
