cmake_minimum_required(VERSION 3.4)
project(rigidTransformation)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED 1)
set(CMAKE_INCLUDE_CURRENT_DIR ON)

find_package(GTest REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(Ceres REQUIRED)

include_directories(${EIGEN3_INCLUDE_DIRS} ${GTEST_INCLUDE_DIRS} ${CERES_INCLUDE_DIRS} include/rigidTransformation)

add_executable(so2_tests tests/so2_tests.cpp)
target_link_libraries(so2_tests ${GTEST_LIBRARIES} ${CERES_LIBRARIES} gtest gtest_main pthread)

add_executable(so3_tests tests/so3_tests.cpp)
target_link_libraries(so3_tests ${GTEST_LIBRARIES} ${CERES_LIBRARIES} gtest gtest_main pthread)

add_executable(se2_tests tests/se2_tests.cpp)
target_link_libraries(se2_tests ${GTEST_LIBRARIES} ${CERES_LIBRARIES} gtest gtest_main pthread)

add_executable(quaternion_tests tests/quaternion_tests.cpp)
target_link_libraries(quaternion_tests ${GTEST_LIBRARIES} ${CERES_LIBRARIES} gtest gtest_main pthread)

add_executable(se3_tests tests/se3_tests.cpp)
target_link_libraries(se3_tests ${GTEST_LIBRARIES} ${CERES_LIBRARIES} gtest gtest_main pthread)
