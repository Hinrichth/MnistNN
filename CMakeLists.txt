cmake_minimum_required(VERSION 3.16)
project(TestNN VERSION 0.1.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_compile_options(-Ofast -mavx -mavx2)

# Project headers
include_directories(
    src
    config
)

# Collect sources
file(GLOB_RECURSE SRC_
     "src/*.cpp"
)

add_executable(TestNN
    main.cpp
    ${SRC_}
)

find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(TestNN PUBLIC OpenMP::OpenMP_CXX)
endif()


# YAML-CPP (FetchContent)
include(FetchContent)

FetchContent_Declare(
    yaml-cpp
    GIT_REPOSITORY https://github.com/jbeder/yaml-cpp.git
    GIT_TAG master
)

FetchContent_MakeAvailable(yaml-cpp)

# Linking YAML to sources
target_link_libraries(TestNN PRIVATE yaml-cpp)
