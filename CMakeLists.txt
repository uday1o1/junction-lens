cmake_minimum_required(VERSION 3.31)

project(JunctionLens VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

include(cmake/dependencies.cmake)
include(CTest)

add_executable(junctionlens-toolchain-probe cpp/tools/toolchain_probe.cpp)

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
endif()

if(BUILD_TESTING)
  add_test(NAME toolchain-probe-json COMMAND junctionlens-toolchain-probe)
  add_test(NAME onnx-probe-help COMMAND junctionlens-onnx-probe --help)
  set_tests_properties(
    toolchain-probe-json
    PROPERTIES
      PASS_REGULAR_EXPRESSION
      "\"cxx_standard\":202002"
  )
endif()
