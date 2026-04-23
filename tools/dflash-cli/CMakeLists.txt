# tools/dflash-cli — DFlash MTP standalone CLI (Phase 1 of the integration).
#
# Sources imported from lucebox-hub/dflash, now self-contained in v5 so we
# can maintain and evolve them alongside ggml-cuda changes.
#
# Phase 2 will lift the decode loop into tools/server as a dflash slot type.

# DFlash is CUDA-only today. Skip the entire subdir when GGML_CUDA is off;
# otherwise we need the CUDA language here (ggml/src/ggml-cuda/ enables it
# in its own scope, but that does not propagate to tools/).
if (NOT GGML_CUDA)
    message(STATUS "llama-dflash: skipping (requires GGML_CUDA=ON)")
    return()
endif()
enable_language(CUDA)
find_package(CUDAToolkit REQUIRED)

# ─── dflash27b static library ──────────────────────────────────────
set(TARGET_LIB dflash27b)
add_library(${TARGET_LIB} STATIC
    errors.cpp
    gguf_target_loader.cpp
    safetensors_draft.cpp
    qwen35_target_graph.cpp
    qwen3_dflash_graph.cpp
    kv_cache.cpp
    f16_convert.cu
    delta_net_chunked.cpp
    session.cpp
)

target_include_directories(${TARGET_LIB}
    PUBLIC  ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(${TARGET_LIB}
    PUBLIC ggml ggml-cuda ggml-base CUDA::cudart
)

# Soften the warning-as-error policy for these ported sources; they were
# written for lucebox-hub's stricter style and produce upstream-aware
# warnings we do not want to gate the entire build on.
if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    target_compile_options(${TARGET_LIB} PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:-Wno-unused-parameter -Wno-unused-function -Wno-unused-variable>
    )
endif()

# ─── llama-dflash CLI binary ───────────────────────────────────────
set(TARGET_BIN llama-dflash)
add_executable(${TARGET_BIN} main.cpp)
target_link_libraries(${TARGET_BIN} PRIVATE ${TARGET_LIB})
target_compile_features(${TARGET_BIN} PRIVATE cxx_std_17)

if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    target_compile_options(${TARGET_BIN} PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:-Wno-unused-parameter -Wno-unused-function -Wno-unused-variable>
    )
endif()

if(LLAMA_TOOLS_INSTALL)
    install(TARGETS ${TARGET_BIN} RUNTIME)
endif()
