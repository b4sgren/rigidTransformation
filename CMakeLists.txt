cmake_minimum_required(VERSION 3.4)
project(rigidTransformation CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED 1)
set(CMAKE_INCLUDE_CURRENT_DIR ON)

option(BUILD_TESTS ON CACHE)
option(BUILD_EXAMPLES ON CACHE)

find_package(Eigen3 REQUIRED)


add_library(${PROJECT_NAME} INTERFACE) # INTERFACE is for header only libs
target_include_directories(${PROJECT_NAME} INTERFACE
  include/rigidTransformation/
  )
target_link_libraries(${PROJECT_NAME} INTERFACE Eigen3::Eigen)

if (BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

if (BUILD_TESTS)
    add_subdirectory(external_libs/googletest)
    add_subdirectory(tests)
endif()
