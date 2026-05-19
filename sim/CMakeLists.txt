cmake_minimum_required(VERSION 3.14)
project(raycaster_simd LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ============================================================================
# Compiler flags: Enable AVX2 + aggressive optimization
# ============================================================================
if(MSVC)
    # MSVC: /arch:AVX2 enables AVX2 intrinsics
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /arch:AVX2 /O2 /fp:fast")
    # Define __AVX2__ for MSVC (it doesn't define it by default)
    add_compile_definitions(__AVX2__)
else()
    # GCC / Clang
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx2 -mfma -O3 -ffast-math")
endif()

# ============================================================================
# Find pybind11
# ============================================================================
# Try config mode first (pip install pybind11), then fallback to subdirectory
find_package(pybind11 CONFIG QUIET)
if(NOT pybind11_FOUND)
    # Fallback: find via Python
    execute_process(
        COMMAND python -m pybind11 --cmakedir
        OUTPUT_VARIABLE PYBIND11_CMAKE_DIR
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    if(PYBIND11_CMAKE_DIR)
        set(pybind11_DIR ${PYBIND11_CMAKE_DIR})
        find_package(pybind11 CONFIG REQUIRED)
    else()
        message(FATAL_ERROR 
            "pybind11 not found. Install with: pip install pybind11\n"
            "Then re-run cmake.")
    endif()
endif()

# ============================================================================
# Build the Python module
# ============================================================================
pybind11_add_module(raycaster_simd
    src/bindings.cpp
    src/raycaster.cpp
)

target_include_directories(raycaster_simd PRIVATE src)

# Install to project root so Python can import it
install(TARGETS raycaster_simd DESTINATION ${CMAKE_SOURCE_DIR}/..)

message(STATUS "=== OmniRay SIMD Raycaster ===")
message(STATUS "Compiler: ${CMAKE_CXX_COMPILER_ID}")
message(STATUS "Flags: ${CMAKE_CXX_FLAGS}")
message(STATUS "pybind11: ${pybind11_VERSION}")
