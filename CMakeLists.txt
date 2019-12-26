cmake_minimum_required (VERSION 2.6)
project (cpp_kalman)
set(CMAKE_CXX_FLAGS "-g -std=gnu++11 ${CMAKE_CXX_FLAGS}")

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

find_package( Eigen3 REQUIRED)
find_package( Boost COMPONENTS program_options REQUIRED )

include_directories(
    src/
    ${Eigen_INCLUDE_DIRS}
    ${Boost_INCLUDE_DIR}
)

add_library(extended_kalman_filter src/ExtendedKalmanFilterStatic.cpp)
target_link_libraries(extended_kalman_filter
    ${EIGEN_LIBRARIES}
)

add_executable(demo src/demo.cpp)
add_executable(tests src/tests.cpp)

target_link_libraries(demo
    extended_kalman_filter
    ${EIGEN_LIBRARIES}
    ${Boost_LIBRARIES}
)

target_link_libraries(tests
    extended_kalman_filter
    ${EIGEN_LIBRARIES}
    ${Boost_LIBRARIES}
)
