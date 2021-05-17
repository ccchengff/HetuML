cmake_minimum_required(VERSION 3.11)

project(HetuML CXX)

include(cmake/config.cmake)
list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake/Modules)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_CXX_FLAGS "-O3 -Wall")

find_package(OpenMP)
if (OPENMP_FOUND)
  message(STATUS "OpenMP Found")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

if(HETU_PARALLEL_ML)
  add_definitions(-DWITH_PARALLEL)
  add_subdirectory(${CMAKE_SOURCE_DIR}/hetuml/cc/ps-lite)
endif()
add_subdirectory(${CMAKE_SOURCE_DIR}/hetuml/cc/ml)
