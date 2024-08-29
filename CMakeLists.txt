cmake_minimum_required(VERSION 3.29)

project(email-classifier CXX CUDA)

set(CMAKE_CXX_STANDARD 20)

set(CMAKE_CUDA_STANDARD 20)

# Settings for multi-configuration generators (Visual Studio)
if(CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_CONFIGURATION_TYPES "Debug;Release;RelWithDebInfo;MinSizeRel" CACHE STRING "Available build configurations" FORCE)
else()
    # Setting for single configuration generators (Makefile, Ninja)
    if(NOT CMAKE_BUILD_TYPE)
        set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build" FORCE)
    endif()
endif()

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    message("Building in Debug mode")
elseif(CMAKE_BUILD_TYPE STREQUAL "Release")
    message("Building in Release mode")
endif()

add_executable(email-classifier
        main.cpp
)

set_target_properties(email-classifier PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
)
