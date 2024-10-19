find_package(Ceres 2.2 REQUIRED)
include_directories(${CERES_INCLUDE_DIRS})

add_executable(so2_example so2_example.cpp)
target_link_libraries(so2_example ${PROJECT_NAME} ${CERES_LIBRARIES})

add_executable(so3_example so3_example.cpp)
target_link_libraries(so3_example ${PROJECT_NAME} ${CERES_LIBRARIES})

add_executable(quaternion_example quaternion_example.cpp)
target_link_libraries(quaternion_example ${PROJECT_NAME} ${CERES_LIBRARIES})

add_executable(se2_example se2_example.cpp)
target_link_libraries(se2_example ${PROJECT_NAME} ${CERES_LIBRARIES})

add_executable(se3_example se3_example.cpp)
target_link_libraries(se3_example ${PROJECT_NAME} ${CERES_LIBRARIES})
