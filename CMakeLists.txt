cmake_minimum_required(VERSION 3.10)

project(VideoStabilization)

find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})

macro(add_example name)
    add_executable(${name} ${name}.cpp)
    target_link_libraries(${name} ${OpenCV_LIBS})
endmacro()

add_example(main)
