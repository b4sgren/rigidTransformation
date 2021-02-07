cmake_minimum_required(VERSION 3.4)
project(rigidTransformation)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED 1)
set(CMAKE_INCLUDE_CURRENT_DIR ON)

add_subdirectory(external_libs/googletest)
find_package(Eigen3 REQUIRED)

include_directories(${EIGEN3_INCLUDE_DIRS} include/rigidTransformation)

add_executable(test_so2 tests/test_so2.cpp)
target_link_libraries(test_so2 gtest gtest_main)

add_executable(test_so3 tests/test_so3.cpp)
target_link_libraries(test_so3 ${GTEST_LIBRARIES} gtest gtest_main pthread)

# add_executable(se2_tests tests/se2_tests.cpp)
# target_link_libraries(se2_tests ${GTEST_LIBRARIES} gtest gtest_main pthread)

add_executable(test_quaternion tests/test_quaternion.cpp)
target_link_libraries(test_quaternion ${GTEST_LIBRARIES} gtest gtest_main pthread)

# add_executable(se3_tests tests/se3_tests.cpp)
# target_link_libraries(se3_tests ${GTEST_LIBRARIES} gtest gtest_main pthread)
