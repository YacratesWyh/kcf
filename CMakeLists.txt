cmake_minimum_required(VERSION 2.8.7)
project(kcftracker C CXX)

find_package(OpenCV REQUIRED)

set(KCFTRACKER_TARGET_VERSION "1.0.0")
set(KCFTRACKER_TARGET_SOVERSION "1.0.0")

if(NOT WIN32)
ADD_DEFINITIONS("-std=c++0x -O3")
endif(NOT WIN32)

set(KCFTRACKER_INCLUDE_DIR ${PROJECT_SOURCE_DIR}/include)
include_directories(${KCFTRACKER_INCLUDE_DIR})

add_subdirectory(src)
add_subdirectory(samples)
