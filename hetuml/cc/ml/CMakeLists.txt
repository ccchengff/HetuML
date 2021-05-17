# compile hetuml

file(GLOB_RECURSE HETUML_SRC src/*.cc)
list(REMOVE_ITEM HETUML_SRC ${CMAKE_CURRENT_SOURCE_DIR}/src/python_binding/api.cc)
if(NOT HETU_PARALLEL_ML)
  list(REMOVE_ITEM HETUML_SRC ${CMAKE_CURRENT_SOURCE_DIR}/src/parallel_ml.cc)
  list(REMOVE_ITEM HETUML_SRC ${CMAKE_CURRENT_SOURCE_DIR}/src/model/lda/parallel_lda.cc)
endif()
add_library(hetuml SHARED ${HETUML_SRC})
target_include_directories(hetuml PUBLIC include)
target_include_directories(hetuml PUBLIC src)

if(HETU_PARALLEL_ML)
  target_include_directories(hetuml PUBLIC ${PS_SRC}/include)
  target_link_libraries(hetuml PUBLIC ps)
endif()

# pybind11
find_package(pybind11)
if(NOT pybind11_FOUND)
  message(STATUS "pybind11 not found, fetching...")
  include(FetchContent)
  FetchContent_Declare(
      pybind11
      GIT_REPOSITORY https://github.com/pybind/pybind11
      GIT_TAG        v2.6.2
  )
  FetchContent_GetProperties(pybind11)
  if(NOT pybind11_POPULATED)
      FetchContent_Populate(pybind11)
      add_subdirectory(${pybind11_SOURCE_DIR} ${pybind11_BINARY_DIR})
  endif()
endif()
pybind11_add_module(hetuml_core src/python_binding/api.cc)
target_include_directories(hetuml_core PUBLIC include)
target_include_directories(hetuml_core PUBLIC src)
target_link_libraries(hetuml_core PRIVATE hetuml)
