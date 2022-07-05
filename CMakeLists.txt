cmake_minimum_required(VERSION 3.4)
project(rigidTransformation CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED 1)
set(CMAKE_INCLUDE_CURRENT_DIR ON)

option(BUILD_TESTS OFF CACHE)
option(BUILD_EXAMPLES OFF CACHE)

find_package(Eigen3 REQUIRED)

# Use following to install/find library
#https://stackoverflow.com/questions/47718485/install-and-export-interface-only-library-cmake?answertab=trending#tab-top
add_library(${PROJECT_NAME} INTERFACE) # INTERFACE is for header only libs
target_include_directories(${PROJECT_NAME} INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
  )
target_link_libraries(${PROJECT_NAME} INTERFACE Eigen3::Eigen)

include(CMakePackageConfigHelpers)
write_basic_package_version_file(
    "${PROJECT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake"
    VERSION 0.1
    COMPATIBILITY AnyNewerVersion
)

set(LIB_DEST lib/${PROJECT_NAME})
set(INCLUDE_DEST include)
install(TARGETS ${PROJECT_NAME}
        EXPORT "${PROJECT_NAME}Targets"
        ARCHIVE DESTINATION ${LIB_DEST} COMPONENT Development
        LIBRARY DESTINATION ${LIB_DEST} COMPONENT Runtime
        RUNTIME DESTINATION bin COMPONENT Runtime
        PUBLIC_HEADER DESTINATION ${INCLUDE_DEST} COMPONENT Development
        BUNDLE DESTINATION bin COMPONENT Runtime
)

include(CMakePackageConfigHelpers)
configure_package_config_file(
    "${PROJECT_SOURCE_DIR}/cmake/${PROJECT_NAME}Config.cmake.in"
    "${PROJECT_BINARY_DIR}/${PROJECT_NAME}Config.cmake"
    INSTALL_DESTINATION "lib/cmake/${PROJECT_NAME}"
)

install(EXPORT "${PROJECT_NAME}Targets" DESTINATION "lib/cmake/${PROJECT_NAME}")
install(FILES "${PROJECT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake"
              "${PROJECT_BINARY_DIR}/${PROJECT_NAME}Config.cmake"
        DESTINATION "lib/cmake/${PROJECT_NAME}"
)
install(DIRECTORY ${PROJECT_SOURCE_DIR}/include/ DESTINATION include)

if (BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

if (BUILD_TESTS)
    add_subdirectory(external_libs/googletest)
    add_subdirectory(tests)
endif()
