cmake_minimum_required(VERSION 3.20)
project(gossip_rl VERSION 1.0.0 LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Build options
option(BUILD_TESTS "Build unit tests" ON)
option(BUILD_BENCHMARKS "Build benchmarks" OFF)
option(USE_URING "Use io_uring (Linux only)" ON)
option(ENABLE_ASAN "Enable AddressSanitizer" OFF)

# Detect platform
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    set(PLATFORM_LINUX TRUE)
    message(STATUS "Building for Linux - io_uring enabled")
elseif(CMAKE_SYSTEM_NAME STREQUAL "Darwin")
    set(PLATFORM_MACOS TRUE)
    set(USE_URING OFF)
    message(STATUS "Building for macOS - using kqueue fallback")
endif()

# Compiler flags
add_compile_options(-Wall -Wextra -Wpedantic)
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    add_compile_options(-O3 -march=native -flto)
endif()

if(ENABLE_ASAN)
    add_compile_options(-fsanitize=address -fno-omit-frame-pointer)
    add_link_options(-fsanitize=address)
endif()

# Dependencies
find_package(Threads REQUIRED)

# FlatBuffers for zero-copy serialization
include(FetchContent)
FetchContent_Declare(
    flatbuffers
    GIT_REPOSITORY https://github.com/google/flatbuffers.git
    GIT_TAG v23.5.26
)
FetchContent_MakeAvailable(flatbuffers)

# spdlog for logging
FetchContent_Declare(
    spdlog
    GIT_REPOSITORY https://github.com/gabime/spdlog.git
    GIT_TAG v1.12.0
)
FetchContent_MakeAvailable(spdlog)

# liburing (Linux only)
if(PLATFORM_LINUX AND USE_URING)
    find_library(URING_LIB uring)
    if(NOT URING_LIB)
        message(STATUS "liburing not found, fetching...")
        FetchContent_Declare(
            liburing
            GIT_REPOSITORY https://github.com/axboe/liburing.git
            GIT_TAG liburing-2.4
        )
        FetchContent_MakeAvailable(liburing)
    endif()
    add_definitions(-DUSE_IO_URING)
endif()

# Include directories
include_directories(${CMAKE_SOURCE_DIR}/cpp/include)

# Subdirectories
add_subdirectory(cpp)

if(BUILD_TESTS)
    enable_testing()
    FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG v1.14.0
    )
    FetchContent_MakeAvailable(googletest)
    add_subdirectory(tests/cpp)
endif()

# Generate version header
configure_file(
    ${CMAKE_SOURCE_DIR}/cpp/include/version.hpp.in
    ${CMAKE_SOURCE_DIR}/cpp/include/version.hpp
)

# Install targets
install(TARGETS gossip_rpc
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
)

install(DIRECTORY cpp/include/
    DESTINATION include/gossip_rl
    FILES_MATCHING PATTERN "*.hpp"
)
