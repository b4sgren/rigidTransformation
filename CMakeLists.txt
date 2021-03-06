cmake_minimum_required(VERSION 3.4)
project(rigidTransformation CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED 1)
set(CMAKE_INCLUDE_CURRENT_DIR ON)

add_subdirectory(external_libs/googletest)
find_package(Eigen3 REQUIRED)

include_directories(${EIGEN3_INCLUDE_DIRS})

add_library(${PROJECT_NAME} INTERFACE) # INTERFACE is for header only libs
target_include_directories(${PROJECT_NAME} INTERFACE
  include/rigidTransformation/
  )

add_executable(test_so2 tests/test_so2.cpp)
target_link_libraries(test_so2 gtest gtest_main ${PROJECT_NAME})

add_executable(test_so3 tests/test_so3.cpp)
target_link_libraries(test_so3 ${PROJECT_NAME} gtest gtest_main)

add_executable(test_se2 tests/test_se2.cpp)
target_link_libraries(test_se2 ${PROJECT_NAME} gtest gtest_main)

add_executable(test_quaternion tests/test_quaternion.cpp)
target_link_libraries(test_quaternion ${PROJECT_NAME} gtest gtest_main)

add_executable(test_se3 tests/test_se3.cpp)
target_link_libraries(test_se3 ${PROJECT_NAME} gtest gtest_main)
