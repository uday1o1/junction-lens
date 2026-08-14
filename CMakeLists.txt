cmake_minimum_required(VERSION 3.31)

project(JunctionLens VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

include(CTest)
include(cmake/dependencies.cmake)
include(cmake/sanitizers.cmake)

add_executable(junctionlens-toolchain-probe cpp/tools/toolchain_probe.cpp)
set_target_properties(
  junctionlens-toolchain-probe
  PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}"
)

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
set_target_properties(
  junctionlens-contract-probe
  PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}"
)

add_library(junctionlens_geometry STATIC cpp/src/geometry/geometry.cpp)
add_library(junctionlens::geometry ALIAS junctionlens_geometry)
target_include_directories(junctionlens_geometry PUBLIC "${PROJECT_SOURCE_DIR}/cpp/include")
target_link_libraries(junctionlens_geometry PUBLIC Eigen3::Eigen)
junctionlens_enable_sanitizers(junctionlens_geometry)

add_library(junctionlens_eval STATIC cpp/src/eval/metrics.cpp)
add_library(junctionlens::eval ALIAS junctionlens_eval)
target_include_directories(junctionlens_eval PUBLIC "${PROJECT_SOURCE_DIR}/cpp/include")
junctionlens_enable_sanitizers(junctionlens_eval)

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
    RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}"
    BUILD_RPATH "${JUNCTIONLENS_ONNXRUNTIME_ROOT}/lib"
    INSTALL_RPATH "$ORIGIN/../lib"
)

add_library(
  junctionlens_infer
  STATIC
    cpp/src/infer/buffer_pool.cpp
    cpp/src/infer/preprocess.cpp
    cpp/src/infer/runtime.cpp
    cpp/src/infer/sha256.cpp
)
add_library(junctionlens::infer ALIAS junctionlens_infer)
target_include_directories(
  junctionlens_infer
  PUBLIC
    "${PROJECT_SOURCE_DIR}/cpp/include"
    "${PROJECT_SOURCE_DIR}/cpp"
)
target_include_directories(
  junctionlens_infer
  SYSTEM PRIVATE
    "${PROJECT_BINARY_DIR}"
    "${junctionlens_opencv_SOURCE_DIR}/modules/core/include"
    "${junctionlens_opencv_SOURCE_DIR}/modules/imgcodecs/include"
    "${junctionlens_opencv_SOURCE_DIR}/modules/imgproc/include"
)
target_link_libraries(
  junctionlens_infer
  PUBLIC
    junctionlens::contract
  PRIVATE
    Eigen3::Eigen
    onnxruntime::onnxruntime
    opencv_core
    opencv_imgcodecs
    opencv_imgproc
)
junctionlens_enable_sanitizers(junctionlens_infer)
set_target_properties(
  junctionlens_infer
  PROPERTIES
    BUILD_RPATH "${JUNCTIONLENS_ONNXRUNTIME_ROOT}/lib"
)

add_executable(junctionlens-runtime cpp/runtime/main.cpp)
target_link_libraries(junctionlens-runtime PRIVATE junctionlens::infer onnxruntime::onnxruntime)
set_target_properties(
  junctionlens-runtime
  PROPERTIES
    BUILD_RPATH "${JUNCTIONLENS_ONNXRUNTIME_ROOT}/lib"
    INSTALL_RPATH "$ORIGIN/../lib"
)
junctionlens_enable_sanitizers(junctionlens-runtime)

if(MSVC)
  target_compile_options(junctionlens-toolchain-probe PRIVATE /W4 /WX)
  target_compile_options(junctionlens-onnx-probe PRIVATE /W4 /WX)
  target_compile_options(junctionlens_infer PRIVATE /W4 /WX)
  target_compile_options(junctionlens-runtime PRIVATE /W4 /WX)
  target_compile_options(junctionlens_contract PRIVATE /W4 /WX)
  target_compile_options(junctionlens-contract-probe PRIVATE /W4 /WX)
  target_compile_options(junctionlens_geometry PRIVATE /W4 /WX)
  target_compile_options(junctionlens_eval PRIVATE /W4 /WX)
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
  target_compile_options(junctionlens_infer PRIVATE ${JUNCTIONLENS_WARNING_OPTIONS})
  target_compile_options(junctionlens-runtime PRIVATE ${JUNCTIONLENS_WARNING_OPTIONS})
  target_compile_options(junctionlens_contract PRIVATE ${JUNCTIONLENS_WARNING_OPTIONS})
  target_compile_options(junctionlens-contract-probe PRIVATE ${JUNCTIONLENS_WARNING_OPTIONS})
  target_compile_options(junctionlens_geometry PRIVATE ${JUNCTIONLENS_WARNING_OPTIONS})
  target_compile_options(junctionlens_eval PRIVATE ${JUNCTIONLENS_WARNING_OPTIONS})
  if(APPLE)
    set(JUNCTIONLENS_APPLE_THIRD_PARTY_WARNINGS -Wno-gcc-compat -Wno-nullability-extension)
    target_compile_options(junctionlens_contract PRIVATE ${JUNCTIONLENS_APPLE_THIRD_PARTY_WARNINGS})
    target_compile_options(junctionlens-contract-probe PRIVATE ${JUNCTIONLENS_APPLE_THIRD_PARTY_WARNINGS})
    target_compile_options(junctionlens_infer PRIVATE ${JUNCTIONLENS_APPLE_THIRD_PARTY_WARNINGS})
    target_compile_options(junctionlens-runtime PRIVATE ${JUNCTIONLENS_APPLE_THIRD_PARTY_WARNINGS})
  endif()
endif()

if(BUILD_TESTING)
  add_test(NAME toolchain-probe-json COMMAND junctionlens-toolchain-probe)
  add_test(NAME onnx-probe-help COMMAND junctionlens-onnx-probe --help)
  add_test(NAME runtime-help COMMAND junctionlens-runtime --help)
  add_test(NAME contract-probe-help COMMAND junctionlens-contract-probe --help)
  add_test(
    NAME contract-golden-cpp
    COMMAND junctionlens-contract-probe
      --input "${PROJECT_SOURCE_DIR}/tests/fixtures/contract/v1/golden.pb"
  )
  add_executable(junctionlens-geometry-tests cpp/tests/geometry/geometry_test.cpp)
  target_link_libraries(junctionlens-geometry-tests PRIVATE junctionlens::geometry GTest::gtest_main)
  junctionlens_enable_sanitizers(junctionlens-geometry-tests)
  if(MSVC)
    target_compile_options(junctionlens-geometry-tests PRIVATE /W4 /WX)
  else()
    target_compile_options(junctionlens-geometry-tests PRIVATE ${JUNCTIONLENS_WARNING_OPTIONS})
  endif()
  include(GoogleTest)
  gtest_discover_tests(junctionlens-geometry-tests TEST_PREFIX geometry.)
  add_executable(junctionlens-eval-tests cpp/tests/eval/metrics_test.cpp)
  target_link_libraries(junctionlens-eval-tests PRIVATE junctionlens::eval GTest::gtest_main)
  target_compile_definitions(
    junctionlens-eval-tests
    PRIVATE
      JUNCTIONLENS_CUSTOM_METRIC_GOLDENS="${PROJECT_SOURCE_DIR}/tests/fixtures/evaluator/custom_metric_goldens.txt"
  )
  junctionlens_enable_sanitizers(junctionlens-eval-tests)
  if(MSVC)
    target_compile_options(junctionlens-eval-tests PRIVATE /W4 /WX)
  else()
    target_compile_options(junctionlens-eval-tests PRIVATE ${JUNCTIONLENS_WARNING_OPTIONS})
  endif()
  gtest_discover_tests(junctionlens-eval-tests TEST_PREFIX eval.)
  add_executable(junctionlens-infer-tests cpp/tests/infer/runtime_test.cpp)
  target_link_libraries(junctionlens-infer-tests PRIVATE junctionlens::infer GTest::gtest_main)
  junctionlens_enable_sanitizers(junctionlens-infer-tests)
  if(MSVC)
    target_compile_options(junctionlens-infer-tests PRIVATE /W4 /WX)
  else()
    target_compile_options(junctionlens-infer-tests PRIVATE ${JUNCTIONLENS_WARNING_OPTIONS})
    if(APPLE)
      target_compile_options(
        junctionlens-infer-tests
        PRIVATE ${JUNCTIONLENS_APPLE_THIRD_PARTY_WARNINGS}
      )
    endif()
  endif()
  gtest_discover_tests(junctionlens-infer-tests TEST_PREFIX infer.)
  set_tests_properties(
    toolchain-probe-json
    PROPERTIES
      PASS_REGULAR_EXPRESSION
      "\"cxx_standard\":202002"
  )
endif()
