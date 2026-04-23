# tools/dflash-server — Phase 2 of the DFlash integration.
#
# Single-binary OpenAI-compatible HTTP server. Spawns `llama-dflash --daemon`
# as a subprocess for inference; owns the tokenizer + chat templates
# in-process via libcommon / libllama (vocab_only load, no GPU weights here).
#
# Phase 3 will replace the subprocess with an in-process dflash27b session
# and wire DFlash into tools/server proper with slot-level prompt caching.

if (NOT GGML_CUDA)
    message(STATUS "llama-dflash-server: skipping (requires GGML_CUDA=ON for llama-dflash daemon)")
    return()
endif()

set(TARGET llama-dflash-server)
add_executable(${TARGET} main.cpp)
target_link_libraries(${TARGET} PRIVATE
    common
    cpp-httplib
    ${CMAKE_THREAD_LIBS_INIT}
)
# Vendored headers live under ${CMAKE_SOURCE_DIR}/vendor and are referenced
# via `<cpp-httplib/...>` / `<nlohmann/...>` across the tree; match that.
target_include_directories(${TARGET} PRIVATE ${CMAKE_SOURCE_DIR}/vendor)
target_compile_features(${TARGET} PRIVATE cxx_std_17)

if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    target_compile_options(${TARGET} PRIVATE
        -Wno-unused-parameter -Wno-unused-function -Wno-unused-variable
    )
endif()

if(LLAMA_TOOLS_INSTALL)
    install(TARGETS ${TARGET} RUNTIME)
endif()
