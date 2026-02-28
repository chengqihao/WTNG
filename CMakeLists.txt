cmake_minimum_required(VERSION 3.10)
project(STKQ)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
include_directories(${PROJECT_SOURCE_DIR}/include)

#OpenMP
find_package(OpenMP REQUIRED)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
else()
    message(FATAL_ERROR "no OpenMP supprot")
endif()

find_package(Boost REQUIRED)
if(Boost_FOUND)
    include_directories(${Boost_INCLUDE_DIRS})
    link_directories(${Boost_LIBRARY_DIRS})
endif()

add_definitions (-std=c++17 -O3 -lboost -march=native -Wall -DINFO)

include_directories(include/third_party/argparse/include)
include_directories(include/third_party/fmt/include)
include_directories(include/third_party/hnswlib)
include_directories(include/third_party/UNIFY)
include_directories(include/third_party/ygg)

add_subdirectory(src)
add_subdirectory(test)
