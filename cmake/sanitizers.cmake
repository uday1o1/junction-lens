function(junctionlens_enable_sanitizers target)
  if(NOT JUNCTIONLENS_ENABLE_SANITIZERS)
    return()
  endif()
  if(MSVC)
    message(FATAL_ERROR "JunctionLens ASan and UBSan preset requires Clang or GCC")
  endif()
  if(NOT CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
    message(FATAL_ERROR "Unsupported sanitizer compiler: ${CMAKE_CXX_COMPILER_ID}")
  endif()
  target_compile_options(
    ${target}
    PRIVATE
      -fsanitize=address,undefined
      -fno-omit-frame-pointer
      -fno-sanitize-recover=all
  )
  target_link_options(
    ${target}
    PRIVATE
      -fsanitize=address,undefined
      -fno-omit-frame-pointer
      -fno-sanitize-recover=all
  )
endfunction()
