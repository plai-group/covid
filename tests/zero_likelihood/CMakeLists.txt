cmake_minimum_required(VERSION 3.1)

project(pyprob_cpp_test VERSION 0.1 LANGUAGES CXX)

set(CMAKE_BUILD_TYPE RelWithDebInfo)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(simulator main.cpp)
target_link_libraries(simulator pyprob_cpp zmq)