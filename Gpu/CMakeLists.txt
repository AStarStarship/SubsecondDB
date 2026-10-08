# Copyright AStarship <https://astarship.net>.
#
# SubsecondDb GPU Vector Database — build.
#
# Two targets:
#   subseconddb-gpu        — CPU build (always). Scheduler + VectorDb seams,
#                            CPU reference search, GPU mode downgrades to CPU
#                            (NoRunner) on a box with no GPU. No CUDA needed.
#   subseconddb-gpu-cuda   — CUDA build (only if a CUDA toolkit + compiler are
#                            found). Compiles CudaRunner.cu with nvcc, defines
#                            SUBSECONDBD_HAS_CUDA, and the seam runs the live
#                            GPU cross-check. Skipped (with a message) when
#                            CUDA is absent, so the CPU target is never blocked.
#
# CrabsTK Data API (optional): pass -DSUBSECONDBD_USE_CRABSTK_DATA=ON and put
# the CrabsTK root on CMAKE_PREFIX_PATH / -DCRABSTK_DIR=... to enable the
# LoadCorpus/SaveCorpus TRecord/TReader/TWriter path. Off by default so the
# core builds standalone.

cmake_minimum_required(VERSION 3.16)
project(SubsecondDbGpu CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ASCIICrabs include path (the _Config.h chain resolves against it).
set(CRABS_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../ASCIICrabs)

# CrabsTK Data API (optional).
option(SUBSECONDBD_USE_CRABSTK_DATA "Enable CrabsTK Data corpus I/O" OFF)
set(CRABSTK_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../CrabsTK CACHE PATH "CrabsTK root")

# ---------------------------------------------------------------------------
# Shared include dirs (CPU + CUDA targets both need these).
# ---------------------------------------------------------------------------
set(GPU_INCLUDES
  ${CMAKE_CURRENT_SOURCE_DIR}/_Seams
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CRABS_DIR}
  ${CRABS_DIR}/_Seams
)

# ---------------------------------------------------------------------------
# CPU target (always builds).
# ---------------------------------------------------------------------------
add_executable(subseconddb-gpu
  _Seams/_Main.cpp
)
target_include_directories(subseconddb-gpu PRIVATE ${GPU_INCLUDES})
target_compile_options(subseconddb-gpu PRIVATE -Wall)

if(SUBSECONDBD_USE_CRABSTK_DATA)
  target_compile_definitions(subseconddb-gpu PRIVATE SUBSECONDBD_USE_CRABSTK_DATA)
  target_include_directories(subseconddb-gpu PRIVATE ${CRABSTK_DIR})
  message(STATUS "SubsecondDbGpu: CrabsTK Data corpus I/O enabled (${CRABSTK_DIR})")
endif()

# ---------------------------------------------------------------------------
# CUDA target (only if a CUDA toolkit + compiler are found).
# ---------------------------------------------------------------------------
# find_package(CUDA) is the classic module (CMake < 3.17) / CUDA::toolkit in
# 3.17+. We probe for nvcc directly so the check works on a fresh box.
find_program(CUDA_NVCC_EXECUTABLE NAMES nvcc)

if(CUDA_NVCC_EXECUTABLE)
  # Enable the CUDA language for this target.
  enable_language(CUDA)
  set(CMAKE_CUDA_STANDARD 17)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)

  add_executable(subseconddb-gpu-cuda
    _Seams/_Main.cpp
    CudaRunner.cu
  )
  target_include_directories(subseconddb-gpu-cuda PRIVATE ${GPU_INCLUDES})
  target_compile_definitions(subseconddb-gpu-cuda PRIVATE SUBSECONDBD_HAS_CUDA)
  target_compile_options(subseconddb-gpu-cuda PRIVATE -Wall)
  # Link the CUDA runtime (bare-metal; no toolkit libs beyond the runtime).
  target_link_libraries(subseconddb-gpu-cuda PRIVATE CUDA::cudart)
  # The .cu compiles with nvcc; the .cpp with the CXX compiler. CMake handles
  # per-language compilation automatically once CUDA is enabled.
  # Pick the architecture from the box (sm_70 V100 + sm_89/sm_80 5060 Ti).
  # Override with -DCMAKE_CUDA_ARCHITECTURES=native on the target box.
  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 70 80 89)
  endif()

  if(SUBSECONDBD_USE_CRABSTK_DATA)
    target_compile_definitions(subseconddb-gpu-cuda PRIVATE SUBSECONDBD_USE_CRABSTK_DATA)
    target_include_directories(subseconddb-gpu-cuda PRIVATE ${CRABSTK_DIR})
  endif()

  message(STATUS "SubsecondDbGpu: CUDA target enabled (nvcc: ${CUDA_NVCC_EXECUTABLE}, arches: ${CMAKE_CUDA_ARCHITECTURES})")
else()
  message(STATUS "SubsecondDbGpu: nvcc not found — skipping CUDA target. "
                 "The CPU target (subseconddb-gpu) still builds and tests; "
                 "GPU mode downgrades to CPU. Build on the CUDA box with a "
                 "toolkit on PATH to get subseconddb-gpu-cuda.")
endif()
