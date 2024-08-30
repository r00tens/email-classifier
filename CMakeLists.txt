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
        utils/CsvFileHandler.cpp
        utils/Timer.cpp
)

set_target_properties(email-classifier PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
)

# Define the path to the training-dataset folder
set(TRAINING_DATASET_DIR "${CMAKE_SOURCE_DIR}/training-dataset")

# Define the destination directory for the training dataset files in the build directory
set(DEST_DIR "${CMAKE_BINARY_DIR}/training-dataset")

# Find all files in the training-dataset folder
file(GLOB_RECURSE TRAINING_DATASET_FILES "${TRAINING_DATASET_DIR}/*")

# Define the timestamp file that will be used to track if training dataset files have been copied
set(COPY_TIMESTAMP_FILE_TRAINING "${CMAKE_BINARY_DIR}/training_dataset_copy.timestamp")

# Create a custom command for copying training dataset files, with dependencies on the source files
add_custom_command(
        OUTPUT ${COPY_TIMESTAMP_FILE_TRAINING}
        COMMAND ${CMAKE_COMMAND} -E copy_directory ${TRAINING_DATASET_DIR} ${DEST_DIR}
        COMMAND ${CMAKE_COMMAND} -E touch ${COPY_TIMESTAMP_FILE_TRAINING}
        DEPENDS ${TRAINING_DATASET_FILES}
        COMMENT "Copying updated training dataset files to build directory..."
)

# Create a custom target for copying the training dataset files
add_custom_target(copy_training_dataset ALL
        DEPENDS ${COPY_TIMESTAMP_FILE_TRAINING}
)

# Define the path to the test-dataset folder
set(TEST_DATASET_DIR "${CMAKE_SOURCE_DIR}/test-dataset")

# Define the destination directory for the test dataset files in the build directory
set(TEST_DEST_DIR "${CMAKE_BINARY_DIR}/test-dataset")

# Find all files in the test-dataset folder
file(GLOB_RECURSE TEST_DATASET_FILES "${TEST_DATASET_DIR}/*")

# Define the timestamp file that will be used to track if test dataset files have been copied
set(COPY_TIMESTAMP_FILE_TEST "${CMAKE_BINARY_DIR}/test_dataset_copy.timestamp")

# Create a custom command for copying test dataset files, with dependencies on the source files
add_custom_command(
        OUTPUT ${COPY_TIMESTAMP_FILE_TEST}
        COMMAND ${CMAKE_COMMAND} -E copy_directory ${TEST_DATASET_DIR} ${TEST_DEST_DIR}
        COMMAND ${CMAKE_COMMAND} -E touch ${COPY_TIMESTAMP_FILE_TEST}
        DEPENDS ${TEST_DATASET_FILES}
        COMMENT "Copying updated test dataset files to build directory..."
)

# Create a custom target for copying the test dataset files
add_custom_target(copy_test_dataset ALL
        DEPENDS ${COPY_TIMESTAMP_FILE_TEST}
)

# Add dependencies between copying the files and building the executable
add_dependencies(email-classifier copy_training_dataset)
add_dependencies(email-classifier copy_test_dataset)
