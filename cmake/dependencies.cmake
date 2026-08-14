include(FetchContent)

set(JUNCTIONLENS_PROTOBUF_VERSION "31.1")
set(JUNCTIONLENS_PROTOBUF_RUNTIME_VERSION "6.31.1")
set(JUNCTIONLENS_ABSEIL_COMMIT "9ac7062b1860d895fb5a8cbf58c3e9ef8f674b5f")
set(JUNCTIONLENS_ONNX_VERSION "1.18.0")
set(JUNCTIONLENS_ONNXRUNTIME_VERSION "1.25.0")
set(JUNCTIONLENS_EIGEN_VERSION "3.4.0")
set(JUNCTIONLENS_OPENCV_VERSION "4.11.0")
set(JUNCTIONLENS_PYBIND11_VERSION "3.0.1")
set(JUNCTIONLENS_GOOGLETEST_VERSION "1.16.0")
set(JUNCTIONLENS_RAPIDCHECK_COMMIT "6e8dadfdafa3a74eabb52ead87f8787f72eccd0b")

set(
  JUNCTIONLENS_ONNXRUNTIME_ROOT
  "${PROJECT_SOURCE_DIR}/.tools/onnxruntime-cpp/${JUNCTIONLENS_ONNXRUNTIME_VERSION}"
  CACHE PATH
  "Path to the exact locked ONNX Runtime C++ release archive"
)

if(APPLE)
  set(_junctionlens_onnxruntime_library "${JUNCTIONLENS_ONNXRUNTIME_ROOT}/lib/libonnxruntime.dylib")
elseif(UNIX)
  set(_junctionlens_onnxruntime_library "${JUNCTIONLENS_ONNXRUNTIME_ROOT}/lib/libonnxruntime.so")
endif()

if(DEFINED _junctionlens_onnxruntime_library AND EXISTS "${_junctionlens_onnxruntime_library}")
  add_library(onnxruntime::onnxruntime SHARED IMPORTED GLOBAL)
  set_target_properties(
    onnxruntime::onnxruntime
    PROPERTIES
      IMPORTED_LOCATION "${_junctionlens_onnxruntime_library}"
      INTERFACE_INCLUDE_DIRECTORIES "${JUNCTIONLENS_ONNXRUNTIME_ROOT}/include"
  )
endif()

FetchContent_Declare(
  absl
  GIT_REPOSITORY https://github.com/abseil/abseil-cpp.git
  GIT_TAG ${JUNCTIONLENS_ABSEIL_COMMIT}
  GIT_SHALLOW FALSE
  EXCLUDE_FROM_ALL
)
FetchContent_Declare(
  junctionlens_protobuf
  URL https://github.com/protocolbuffers/protobuf/releases/download/v31.1/protobuf-31.1.tar.gz
  URL_HASH SHA256=12bfd76d27b9ac3d65c00966901609e020481b9474ef75c7ff4601ac06fa0b82
  DOWNLOAD_EXTRACT_TIMESTAMP TRUE
  EXCLUDE_FROM_ALL
)

set(protobuf_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(protobuf_BUILD_PROTOC_BINARIES OFF CACHE BOOL "" FORCE)
set(protobuf_BUILD_LIBPROTOC OFF CACHE BOOL "" FORCE)
set(protobuf_BUILD_LIBUPB OFF CACHE BOOL "" FORCE)
set(protobuf_BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(protobuf_INSTALL OFF CACHE BOOL "" FORCE)
set(utf8_range_ENABLE_INSTALL OFF CACHE BOOL "" FORCE)
set(ABSL_ENABLE_INSTALL OFF CACHE BOOL "" FORCE)
set(ABSL_PROPAGATE_CXX_STD ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(junctionlens_protobuf)
FetchContent_Declare(
  junctionlens_eigen
  URL https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz
  URL_HASH SHA256=8586084f71f9bde545ee7fa6d00288b264a2b7ac3607b974e54d13e7162c1c72
  DOWNLOAD_EXTRACT_TIMESTAMP TRUE
  EXCLUDE_FROM_ALL
)
FetchContent_Declare(
  junctionlens_opencv
  URL https://github.com/opencv/opencv/archive/refs/tags/4.11.0.tar.gz
  URL_HASH SHA256=9a7c11f924eff5f8d8070e297b322ee68b9227e003fd600d4b8122198091665f
  DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)
FetchContent_Declare(
  junctionlens_pybind11
  URL https://github.com/pybind/pybind11/archive/refs/tags/v3.0.1.tar.gz
  URL_HASH SHA256=741633da746b7c738bb71f1854f957b9da660bcd2dce68d71949037f0969d0ca
  DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)
FetchContent_Declare(
  junctionlens_googletest
  URL https://github.com/google/googletest/archive/refs/tags/v1.16.0.tar.gz
  URL_HASH SHA256=78c676fc63881529bf97bf9d45948d905a66833fbfa5318ea2cd7478cb98f399
  DOWNLOAD_EXTRACT_TIMESTAMP TRUE
  EXCLUDE_FROM_ALL
)
FetchContent_Declare(
  junctionlens_rapidcheck
  URL https://github.com/emil-e/rapidcheck/archive/6e8dadfdafa3a74eabb52ead87f8787f72eccd0b.tar.gz
  URL_HASH SHA256=7ff6fb341db865f4e8e451582627f854f24c8e4d3d7301bd6c6e4ff16f885ba1
  DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)

set(EIGEN_BUILD_DOC OFF CACHE BOOL "" FORCE)
set(_junctionlens_build_testing "${BUILD_TESTING}")
set(BUILD_TESTING OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(junctionlens_eigen)
# Eigen does not rewrite its generated CTest registry when BUILD_TESTING changes
# from ON to OFF in an existing build tree. Remove that stale generated file so
# CTest cannot discover excluded third-party tests after an incremental configure.
file(REMOVE "${junctionlens_eigen_BINARY_DIR}/CTestTestfile.cmake")
get_target_property(_junctionlens_eigen_includes eigen INTERFACE_INCLUDE_DIRECTORIES)
set_property(
  TARGET eigen
  PROPERTY INTERFACE_SYSTEM_INCLUDE_DIRECTORIES "${_junctionlens_eigen_includes}"
)
unset(_junctionlens_eigen_includes)
set(BUILD_TESTING "${_junctionlens_build_testing}" CACHE BOOL "" FORCE)
unset(_junctionlens_build_testing)

if(BUILD_TESTING)
  set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
  set(BUILD_GMOCK OFF CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(junctionlens_googletest)
endif()

# The ONNX Runtime archive is locked for source identity and audit only.
# A build must use the exact Git commit and verify every submodule in containers/images.lock.
FetchContent_Declare(
  junctionlens_onnxruntime_source_identity
  URL https://github.com/microsoft/onnxruntime/archive/refs/tags/v1.25.0.tar.gz
  URL_HASH SHA256=071d31a593082dd1d3a8ff3db8a78fba4e3d841653e225e807b1eb709da5f56f
  DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)
