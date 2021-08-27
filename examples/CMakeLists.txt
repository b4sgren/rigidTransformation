find_package(Ceres REQUIRED)
include_directories(${CERES_INCLUDE_DIRS})

add_executable(so2_example so2_example.cpp)
target_link_libraries(so2_example ${PROJECT_NAME} ${CERES_LIBRARIES})
