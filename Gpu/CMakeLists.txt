cmake_minimum_required(VERSION 3.16)
project(SubsecondDbGpu CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ASCIICrabs include path (the _Config.h chain resolves against it).
set(CRABS_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../ASCIICrabs)

add_executable(subseconddb-gpu
  _Seams/_Main.cpp
)

target_include_directories(subseconddb-gpu PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/_Seams
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CRABS_DIR}
  ${CRABS_DIR}/_Seams
)

# The _Config.h include uses <_Config.h> (angle brackets), so the _Seams dir
# must be on the include path before CRABS (which has its own _Config.h).
target_compile_options(subseconddb-gpu PRIVATE -Wall)
