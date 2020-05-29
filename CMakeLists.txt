cmake_minimum_required(VERSION 3.4)
project(rigidTransformation)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED 1)
set(CMAKE_INCLUDE_CURRENT_DIR ON)

find_package(GTest REQUIRED)
find_package(Eigen3 REQUIRED)

include_directories(${EIGEN3_INCLUDE_DIRS} ${GTEST_INCLUDE_DIRS} include/rigidTransformation)

add_executable(so2_test tests/so2_tests.cpp)
target_link_libraries(so2_test ${GTEST_LIBRARIES} gtest gtest_main)
