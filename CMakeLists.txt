cmake_minimum_required(VERSION 3.21)

# -------------------------
# Policy setup
# -------------------------
if (POLICY CMP0141)
    cmake_policy(SET CMP0141 NEW)

    set(CMAKE_MSVC_DEBUG_INFORMATION_FORMAT
        "$<IF:$<C_COMPILER_ID:MSVC>,$<$<CONFIG:Debug,RelWithDebInfo>:EditAndContinue>,$<$<CONFIG:Debug,RelWithDebInfo>:ProgramDatabase>>"
    )
endif()

# -------------------------
# Force Clang on Windows
# -------------------------
# MSVC C11 atomics are incomplete; prefer clang-cl
if (WIN32)

    if (CMAKE_GENERATOR MATCHES "Visual Studio")
        if (NOT CMAKE_GENERATOR_TOOLSET MATCHES "ClangCL")
            set(CMAKE_GENERATOR_TOOLSET "ClangCL"
                CACHE STRING "Force ClangCL toolset" FORCE
            )
        endif()

    elseif (NOT DEFINED CMAKE_C_COMPILER)
        find_program(CLANG_C_BIN NAMES clang-cl clang)

        if (CLANG_C_BIN)
            set(CMAKE_C_COMPILER "${CLANG_C_BIN}"
                CACHE FILEPATH "Force clang on Windows" FORCE
            )
        else()
            message(FATAL_ERROR
                "clang/clang-cl not found in PATH. Install LLVM or pass -T ClangCL."
            )
        endif()
    endif()

endif()

# -------------------------
# Project
# -------------------------
project(ecs C CXX)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# -------------------------
# mimalloc (default allocator, all platforms)
# -------------------------
include(FetchContent)
set(MI_OVERRIDE        OFF CACHE BOOL "" FORCE)
set(MI_BUILD_SHARED    OFF CACHE BOOL "" FORCE)
set(MI_BUILD_OBJECT    OFF CACHE BOOL "" FORCE)
set(MI_BUILD_TESTS     OFF CACHE BOOL "" FORCE)
FetchContent_Declare(
    mimalloc
    GIT_REPOSITORY https://github.com/microsoft/mimalloc.git
    GIT_TAG        v2.1.7
)
FetchContent_MakeAvailable(mimalloc)

# -------------------------
# zstd (static, used by serialize tests)
# -------------------------
set(ZSTD_BUILD_PROGRAMS OFF CACHE BOOL "" FORCE)
set(ZSTD_BUILD_SHARED   OFF CACHE BOOL "" FORCE)
set(ZSTD_BUILD_STATIC   ON  CACHE BOOL "" FORCE)
set(ZSTD_BUILD_TESTS    OFF CACHE BOOL "" FORCE)
set(ZSTD_LEGACY_SUPPORT OFF CACHE BOOL "" FORCE)
FetchContent_Declare(
    zstd
    GIT_REPOSITORY https://github.com/facebook/zstd.git
    GIT_TAG        v1.5.6
    SOURCE_SUBDIR  build/cmake
)
FetchContent_MakeAvailable(zstd)

# -------------------------
# Target selection (gated by CMAKE_BUILD_TYPE — single-config generator)
#   Debug    → ecs       (tests, sanitizers, asserts)
#   Release  → ecs_bench (Google Benchmark only, no test runtime)
# Override either side with -DECS_FORCE_BUILD_TESTS=ON / -DECS_FORCE_BUILD_BENCH=ON.
# -------------------------
if (NOT DEFINED CMAKE_BUILD_TYPE OR CMAKE_BUILD_TYPE STREQUAL "")
    set(CMAKE_BUILD_TYPE "Debug" CACHE STRING "" FORCE)
endif()

option(ECS_FORCE_BUILD_TESTS "Force-build the ecs (tests) target" OFF)
option(ECS_FORCE_BUILD_BENCH "Force-build the ecs_bench target"   OFF)

if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(ECS_BUILD_TESTS TRUE)
    set(ECS_BUILD_BENCH ${ECS_FORCE_BUILD_BENCH})
else()
    set(ECS_BUILD_TESTS ${ECS_FORCE_BUILD_TESTS})
    set(ECS_BUILD_BENCH TRUE)
endif()

# -------------------------
# Executables
# -------------------------
if (ECS_BUILD_TESTS)
    add_executable(ecs
        main.c
        src/ecs.c
        src/ecs_query.c
        src/broadphase.c
        src/input.c
    )

    target_include_directories(ecs PRIVATE
        src
        tests
        ${zstd_SOURCE_DIR}/lib
    )

    target_link_libraries(ecs PRIVATE mimalloc-static libzstd_static)
endif()

if (ECS_BUILD_BENCH)
    set(BENCHMARK_ENABLE_TESTING     OFF CACHE BOOL "" FORCE)
    set(BENCHMARK_ENABLE_GTEST_TESTS OFF CACHE BOOL "" FORCE)
    set(BENCHMARK_ENABLE_INSTALL     OFF CACHE BOOL "" FORCE)
    set(BENCHMARK_ENABLE_WERROR      OFF CACHE BOOL "" FORCE)
    FetchContent_Declare(
        googlebenchmark
        GIT_REPOSITORY https://github.com/google/benchmark.git
        GIT_TAG        v1.9.0
    )
    FetchContent_MakeAvailable(googlebenchmark)

    add_executable(ecs_bench
        benchmarks/bench_main.cpp
        benchmarks/bench_integrate.c
        benchmarks/bench_broadphase.c
        src/ecs.c
        src/ecs_query.c
        src/broadphase.c
    )

    target_include_directories(ecs_bench PRIVATE
        src
        benchmarks
    )

    target_link_libraries(ecs_bench PRIVATE mimalloc-static benchmark::benchmark)
endif()

# -------------------------
# Compiler options
# -------------------------
# fixed.h SIMD backend selects via arch macros: x86 → SSE4.1+AVX2, arm64 → NEON,
# else scalar. x86-only flags (/arch:AVX2, -msse4.1, -mavx2) must be gated.
if (CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64|amd64|i.86|x86)$")
    set(ECS_ARCH_X86 TRUE)
else()
    set(ECS_ARCH_X86 FALSE)
endif()

set(_ecs_targets)
if (ECS_BUILD_TESTS)
    list(APPEND _ecs_targets ecs)
endif()
if (ECS_BUILD_BENCH)
    list(APPEND _ecs_targets ecs_bench)
endif()

foreach(_tgt ${_ecs_targets})
    if (CMAKE_C_COMPILER_ID STREQUAL "Clang"
        AND CMAKE_C_COMPILER_FRONTEND_VARIANT STREQUAL "MSVC")

        # clang-cl (MSVC-style flags). /arch:AVX2 enables SSE4.1 + AVX2 intrinsics.
        if (ECS_ARCH_X86)
            target_compile_options(${_tgt} PRIVATE /arch:AVX2)
        endif()

        target_compile_options(${_tgt} PRIVATE
            $<$<CONFIG:Release>:/O2 /DNDEBUG>
            $<$<CONFIG:RelWithDebInfo>:/O2 /DNDEBUG>
        )

    elseif (MSVC)

        target_compile_options(${_tgt} PRIVATE
            $<$<COMPILE_LANGUAGE:C>:/experimental:c11atomics>
        )
        if (ECS_ARCH_X86)
            target_compile_options(${_tgt} PRIVATE /arch:AVX2)
        endif()

        target_compile_options(${_tgt} PRIVATE
            $<$<CONFIG:Release>:/O2 /Oi /GL /DNDEBUG>
            $<$<CONFIG:RelWithDebInfo>:/O2 /Oi /GL /DNDEBUG>
        )

        target_link_options(${_tgt} PRIVATE
            $<$<CONFIG:Release>:/LTCG>
            $<$<CONFIG:RelWithDebInfo>:/LTCG>
        )

    else()

        # GCC / Clang (GNU-style). x86 Debug needs -msse4.1 -mavx2 to enable intrinsics;
        # arm64 / other archs use the scalar or NEON backend with no extra flags.
        # Release uses -march=native everywhere — picks up whatever the host supports.
        if (ECS_ARCH_X86)
            target_compile_options(${_tgt} PRIVATE
                $<$<CONFIG:Debug>:-msse4.1 -mavx2>
            )
        endif()

        target_compile_options(${_tgt} PRIVATE
            $<$<CONFIG:Release>:-O3 -flto -march=native -DNDEBUG>
            $<$<CONFIG:RelWithDebInfo>:-O3 -flto -march=native -DNDEBUG>
        )

        target_link_options(${_tgt} PRIVATE
            $<$<CONFIG:Release>:-flto>
            $<$<CONFIG:RelWithDebInfo>:-flto>
        )

    endif()
endforeach()

