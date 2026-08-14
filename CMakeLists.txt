cmake_minimum_required(VERSION 3.31)

project(JunctionLens VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

include(cmake/dependencies.cmake)
include(CTest)

add_executable(junctionlens-toolchain-probe cpp/tools/toolchain_probe.cpp)

if(MSVC)
  target_compile_options(junctionlens-toolchain-probe PRIVATE /W4 /WX)
else()
  target_compile_options(
    junctionlens-toolchain-probe
    PRIVATE
      -Wall
      -Wextra
      -Wpedantic
      -Wconversion
      -Wshadow
      -Werror
  )
endif()

if(BUILD_TESTING)
  add_test(NAME toolchain-probe-json COMMAND junctionlens-toolchain-probe)
  set_tests_properties(
    toolchain-probe-json
    PROPERTIES
      PASS_REGULAR_EXPRESSION
      "\"cxx_standard\":202002"
  )
endif()
