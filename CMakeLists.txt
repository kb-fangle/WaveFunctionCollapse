cmake_minimum_required(VERSION 3.17)
project(WaveFunctionCollapse VERSION 0.1 LANGUAGES C)
cmake_policy(SET CMP0009 NEW) # Do not follow symlinks with GLOB_RECURSE

include(CheckLanguage)

find_package(OpenMP REQUIRED)


STRING(COMPARE EQUAL "${CMAKE_SOURCE_DIR}" "${CMAKE_BINARY_DIR}" insource)
GET_FILENAME_COMPONENT(PARENTDIR ${CMAKE_SOURCE_DIR} PATH)
STRING(COMPARE EQUAL "${CMAKE_SOURCE_DIR}" "${PARENTDIR}" insourcesubdir)
if(insource OR insourcesubdir)
    MESSAGE(FATAL_ERROR "${CMAKE_PROJECT_NAME} requires an out of source build")
endif()

# if(CMAKE_BUILD_TYPE STREQUAL "Release" OR CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo")
#     set(CMAKE_BUILD_TYPE RelWithDebInfo)
# else()
#     set(CMAKE_BUILD_TYPE Debug)
# endif()

set(CMAKE_POSITION_INDEPENDENT_CODE ON) # Pass -fPIC
set(CMAKE_COLOR_MAKEFILE            ON)
set(CMAKE_COLOR_DIAGNOSTICS         ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS   ON) # Always to that...
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

file(GLOB_RECURSE WaveFunctionCollapse_SRC CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/*.c")
file(GLOB_RECURSE WaveFunctionCollapse_INC CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/inc/*.h")

add_executable(wfc ${WaveFunctionCollapse_SRC} ${WaveFunctionCollapse_INC})
target_include_directories(wfc PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/inc)
target_link_libraries(wfc PRIVATE OpenMP::OpenMP_C m)
target_compile_options(wfc PRIVATE -march=native -std=c17)

check_language(CUDA)
if (CMAKE_CUDA_COMPILER)
    enable_language(CUDA)

    file(GLOB_RECURSE WaveFunctionCollapse_CUDA_SRC CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cu")
    add_library(cuda_solver OBJECT ${WaveFunctionCollapse_CUDA_SRC})
    target_include_directories(cuda_solver PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/inc)
    target_compile_definitions(cuda_solver PRIVATE WFC_CUDA CUDA_FORCE_CDP1_IF_SUPPORTED)
    target_compile_options(cuda_solver PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-rdc=true>)

    target_compile_definitions(wfc PRIVATE WFC_CUDA)
    target_link_libraries(wfc PRIVATE cuda_solver)
else()
    message(STATUS "No CUDA support")
endif()

if (MAQAO)
    target_compile_options(wfc PRIVATE -g -fno-omit-frame-pointer)
endif()

target_compile_options(wfc PRIVATE
    -Wall -Wextra -pedantic #-Werror
    -Wshadow
    -Wcast-align -Wconversion -Wsign-conversion -Wdouble-promotion
    # -Wunused-variable
    -Wmisleading-indentation
    -Wnull-dereference
    -Wformat=2
)

if(${CMAKE_C_COMPILER_ID} STREQUAL "Clang")
    target_compile_options(wfc PRIVATE
        -Weverything
        -Wno-vla
        -Wno-unsafe-buffer-usage
        -Wno-declaration-after-statement
        -Wno-documentation
    )
elseif(${CMAKE_C_COMPILER_ID} STREQUAL "GNU")
    target_compile_options(wfc PRIVATE
        -Wno-switch
        -Wno-unsafe-buffer-usage
    )
endif()
