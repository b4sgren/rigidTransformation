#add_executable(test_so2 test_so2.cpp)
#target_link_libraries(test_so2 gtest gtest_main ${PROJECT_NAME})
#
#add_executable(test_so3 test_so3.cpp)
#target_link_libraries(test_so3 ${PROJECT_NAME} gtest gtest_main)
#
#add_executable(test_se2 test_se2.cpp)
#target_link_libraries(test_se2 ${PROJECT_NAME} gtest gtest_main)
#
#add_executable(test_quaternion test_quaternion.cpp)
#target_link_libraries(test_quaternion ${PROJECT_NAME} gtest gtest_main)
#
#add_executable(test_se3 test_se3.cpp)
#target_link_libraries(test_se3 ${PROJECT_NAME} gtest gtest_main)

add_executable(test_uncertain test_uncertain_transforms.cpp)
target_link_libraries(test_uncertain ${PROJECT_NAME} gtest gtest_main)
