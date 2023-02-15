cmake_minimum_required(VERSION 3.13)
project(cv-tutorial-cpp)

find_package(OpenCV REQUIRED)
find_package(OpenVINO REQUIRED)
find_package(NumCpp REQUIRED)

include_directories(
    ${OpenCV_INCLUDE_DIRS}
)

add_executable(
    opencv.out
    opencv.cpp
)
target_link_libraries(
    opencv.out
    ${OpenCV_LIBS}
)

add_executable(
    openvino.out
    openvino.cpp
)
target_link_libraries(
    openvino.out
    ${OpenCV_LIBS}
    openvino::runtime
    NumCpp::NumCpp
)