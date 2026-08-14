cmake_minimum_required(VERSION 3.18)

project(supernodal_gpu LANGUAGES CXX CUDA)

find_package(CUDAToolkit REQUIRED)
find_package(Threads REQUIRED)
find_package(OpenMP QUIET)

find_path(METIS_INCLUDE_DIR metis.h)
find_library(METIS_LIBRARY NAMES metis)
find_path(AMD_INCLUDE_DIR amd.h PATH_SUFFIXES suitesparse)
find_library(AMD_LIBRARY NAMES amd)
find_path(CHOLMOD_INCLUDE_DIR cholmod.h PATH_SUFFIXES suitesparse)
find_library(CHOLMOD_LIBRARY NAMES cholmod)
find_library(SUITESPARSE_CONFIG_LIBRARY NAMES suitesparseconfig SuiteSparseConfig)

# CUDA-enabled SuiteSparse installs include CUDA public headers from cholmod.h.
# Propagate the include directory even though cholmod_symbolic.cpp is C++.
set(CHOLMOD_CUDA_INCLUDE_DIRS ${CUDAToolkit_INCLUDE_DIRS})

if(NOT METIS_INCLUDE_DIR OR NOT METIS_LIBRARY)
    message(FATAL_ERROR
        "METIS was not found. Install libmetis-dev, or set "
        "METIS_INCLUDE_DIR and METIS_LIBRARY when configuring.")
endif()

if(NOT AMD_INCLUDE_DIR OR NOT AMD_LIBRARY)
    message(FATAL_ERROR
        "SuiteSparse AMD was not found. Install libsuitesparse-dev, or set "
        "AMD_INCLUDE_DIR and AMD_LIBRARY when configuring.")
endif()

if(NOT CHOLMOD_INCLUDE_DIR OR NOT CHOLMOD_LIBRARY)
    message(FATAL_ERROR
        "SuiteSparse CHOLMOD was not found. Install libsuitesparse-dev, or set "
        "CHOLMOD_INCLUDE_DIR and CHOLMOD_LIBRARY when configuring.")
endif()

add_library(io_func STATIC
    utils/io_func.cpp
)
target_include_directories(io_func PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_features(io_func PUBLIC cxx_std_11)

add_library(metis_ordering STATIC
    src/metis_ordering.cpp
)
target_include_directories(metis_ordering
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include
    PRIVATE ${METIS_INCLUDE_DIR}
)
target_link_libraries(metis_ordering PUBLIC ${METIS_LIBRARY})
target_compile_features(metis_ordering PUBLIC cxx_std_11)

add_library(amd_ordering STATIC
    src/amd_ordering.cpp
)
target_include_directories(amd_ordering
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include
    PRIVATE ${AMD_INCLUDE_DIR}
)
target_link_libraries(amd_ordering PUBLIC ${AMD_LIBRARY})
if(SUITESPARSE_CONFIG_LIBRARY)
    target_link_libraries(amd_ordering PUBLIC ${SUITESPARSE_CONFIG_LIBRARY})
endif()
target_compile_features(amd_ordering PUBLIC cxx_std_11)

add_library(symbolic_analysis STATIC
    src/fill_in.cpp
)
target_include_directories(symbolic_analysis PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_features(symbolic_analysis PUBLIC cxx_std_11)

add_library(ordering STATIC
    src/ordering.cpp
)
target_include_directories(ordering PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(ordering
    PUBLIC metis_ordering amd_ordering symbolic_analysis
)
if(OpenMP_CXX_FOUND)
    target_link_libraries(ordering PUBLIC OpenMP::OpenMP_CXX)
endif()
target_compile_features(ordering PUBLIC cxx_std_11)

add_library(cholmod_symbolic STATIC
    src/cholmod_symbolic.cpp
)
target_include_directories(cholmod_symbolic
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include
    PRIVATE ${CHOLMOD_INCLUDE_DIR} ${CHOLMOD_CUDA_INCLUDE_DIRS}
)
target_link_libraries(cholmod_symbolic PUBLIC ${CHOLMOD_LIBRARY})
if(SUITESPARSE_CONFIG_LIBRARY)
    target_link_libraries(cholmod_symbolic PUBLIC ${SUITESPARSE_CONFIG_LIBRARY})
endif()
target_link_libraries(cholmod_symbolic PUBLIC CUDA::cublas)
target_compile_features(cholmod_symbolic PUBLIC cxx_std_11)

add_library(gpu_supernodal_ldlt STATIC
    src/gpu_supernodal_ldlt.cu
)
target_include_directories(gpu_supernodal_ldlt
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_link_libraries(gpu_supernodal_ldlt
    PUBLIC CUDA::cudart CUDA::cublas Threads::Threads
)
target_compile_features(gpu_supernodal_ldlt PUBLIC cxx_std_14 cuda_std_14)
set_target_properties(gpu_supernodal_ldlt PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

add_library(unsymmetric_matching STATIC
    src/unsymmetric_matching.cpp
)
target_include_directories(unsymmetric_matching
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_compile_features(unsymmetric_matching PUBLIC cxx_std_14)

add_library(unsymmetric_ordering STATIC
    src/unsymmetric_ordering.cpp
)
target_include_directories(unsymmetric_ordering
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_link_libraries(unsymmetric_ordering
    PUBLIC unsymmetric_matching amd_ordering
)
target_compile_features(unsymmetric_ordering PUBLIC cxx_std_14)

add_library(unsymmetric_symbolic STATIC
    src/unsymmetric_symbolic.cpp
)
target_include_directories(unsymmetric_symbolic
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_compile_features(unsymmetric_symbolic PUBLIC cxx_std_14)

add_library(unsymmetric_verification STATIC
    utils/unsymmetric_verification.cpp
)
target_include_directories(unsymmetric_verification
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_compile_features(unsymmetric_verification PUBLIC cxx_std_14)

add_library(gpu_supernodal_lu STATIC
    src/gpu_supernodal_lu.cu
)
target_include_directories(gpu_supernodal_lu
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src
)
target_link_libraries(gpu_supernodal_lu
    PUBLIC CUDA::cudart CUDA::cublas Threads::Threads unsymmetric_symbolic
)
target_compile_features(gpu_supernodal_lu PUBLIC cxx_std_14 cuda_std_14)
set_target_properties(gpu_supernodal_lu PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

add_executable(supernodal_gpu
    main.cpp
    src/supernodal_app.cpp
    src/unsymmetric_app.cpp
    src/symbolic_pipeline.cpp
    utils/solution_verification.cpp
    utils/symbolic_cache.cpp
)
target_link_libraries(supernodal_gpu
    PRIVATE
        io_func ordering cholmod_symbolic gpu_supernodal_ldlt
        unsymmetric_ordering unsymmetric_symbolic
        unsymmetric_verification gpu_supernodal_lu
)
target_compile_features(supernodal_gpu PRIVATE cxx_std_14)

option(SUPERNODAL_BUILD_TESTS "Build synthetic GPU factorization smoke tests" OFF)
if(SUPERNODAL_BUILD_TESTS)
    enable_testing()
    add_executable(gpu_ldlt_smoke tests/gpu_ldlt_smoke.cu)
    target_link_libraries(gpu_ldlt_smoke PRIVATE gpu_supernodal_ldlt)
    target_compile_features(gpu_ldlt_smoke PRIVATE cxx_std_14 cuda_std_14)
    add_test(NAME gpu_ldlt_smoke COMMAND gpu_ldlt_smoke)

    add_executable(gpu_lu_smoke tests/gpu_lu_smoke.cu)
    target_link_libraries(gpu_lu_smoke
        PRIVATE
            gpu_supernodal_lu unsymmetric_ordering unsymmetric_symbolic)
    target_compile_features(gpu_lu_smoke PRIVATE cxx_std_14 cuda_std_14)
    add_test(NAME gpu_lu_smoke COMMAND gpu_lu_smoke)

    add_executable(unsymmetric_symbolic_smoke
        tests/unsymmetric_symbolic_smoke.cpp)
    target_link_libraries(unsymmetric_symbolic_smoke
        PRIVATE unsymmetric_ordering unsymmetric_symbolic)
    target_compile_features(unsymmetric_symbolic_smoke PRIVATE cxx_std_14)
    add_test(NAME unsymmetric_symbolic_smoke COMMAND unsymmetric_symbolic_smoke)

    add_executable(unsymmetric_matching_smoke
        tests/unsymmetric_matching_smoke.cpp)
    target_link_libraries(unsymmetric_matching_smoke
        PRIVATE unsymmetric_matching)
    target_compile_features(unsymmetric_matching_smoke PRIVATE cxx_std_14)
    add_test(NAME unsymmetric_matching_smoke COMMAND unsymmetric_matching_smoke)
endif()

set(SYMBOLIC_VISUALIZATION_DIR "${CMAKE_BINARY_DIR}/visualization")
file(MAKE_DIRECTORY "${SYMBOLIC_VISUALIZATION_DIR}")
target_compile_definitions(supernodal_gpu PRIVATE
    SUPERNODAL_VISUALIZATION_DIR="${SYMBOLIC_VISUALIZATION_DIR}")

# 在构建目录中生成运行脚本，统一记录程序输出和运行时间。
file(WRITE "${CMAKE_BINARY_DIR}/RunAndLog.cmake" [=[
if(NOT DEFINED EXECUTABLE OR NOT DEFINED WORKING_DIRECTORY OR NOT DEFINED LOG_FILE)
    message(FATAL_ERROR "EXECUTABLE, WORKING_DIRECTORY and LOG_FILE are required")
endif()

string(TIMESTAMP start_time "%Y-%m-%d %H:%M:%S")
string(TIMESTAMP start_epoch "%s")

execute_process(
    COMMAND "${EXECUTABLE}"
    WORKING_DIRECTORY "${WORKING_DIRECTORY}"
    OUTPUT_VARIABLE program_stdout
    ERROR_VARIABLE program_stderr
    RESULT_VARIABLE program_result
)

string(TIMESTAMP end_time "%Y-%m-%d %H:%M:%S")
string(TIMESTAMP end_epoch "%s")
math(EXPR elapsed_seconds "${end_epoch} - ${start_epoch}")

file(APPEND "${LOG_FILE}"
    "[${start_time}] Program started\n"
    "${program_stdout}"
    "${program_stderr}"
    "[${end_time}] Program finished (exit code: ${program_result}, elapsed: ${elapsed_seconds} s)\n\n"
)

message(STATUS "Program output written to: ${LOG_FILE}")

if(NOT program_result EQUAL 0)
    message(FATAL_ERROR "Program exited with code ${program_result}; see ${LOG_FILE}")
endif()
]=])

# 编译并运行统一排序程序。
add_custom_target(run
    COMMAND ${CMAKE_COMMAND}
        "-DEXECUTABLE=$<TARGET_FILE:supernodal_gpu>"
        "-DWORKING_DIRECTORY=${CMAKE_CURRENT_SOURCE_DIR}"
        "-DLOG_FILE=${CMAKE_BINARY_DIR}/supernodal_gpu.log"
        -P "${CMAKE_BINARY_DIR}/RunAndLog.cmake"
    DEPENDS supernodal_gpu
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    USES_TERMINAL
)

find_package(Python3 COMPONENTS Interpreter QUIET)
if(Python3_Interpreter_FOUND)
    add_custom_target(visualize
        COMMAND $<TARGET_FILE:supernodal_gpu> --export-visualization
        COMMAND ${Python3_EXECUTABLE}
            "${CMAKE_CURRENT_SOURCE_DIR}/utils/plot_symbolic_analysis.py"
            --input-dir "${SYMBOLIC_VISUALIZATION_DIR}"
            --output-dir "${SYMBOLIC_VISUALIZATION_DIR}"
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        USES_TERMINAL
    )
    add_dependencies(visualize supernodal_gpu)
else()
    message(WARNING
        "Python 3 was not found; the visualize target will not be available.")
endif()

set_property(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    PROPERTY VS_STARTUP_PROJECT supernodal_gpu
)

set(PROJECT_WARNING_TARGETS
    io_func
    metis_ordering
    amd_ordering
    symbolic_analysis
    ordering
    cholmod_symbolic
    gpu_supernodal_ldlt
    unsymmetric_matching
    unsymmetric_ordering
    unsymmetric_symbolic
    unsymmetric_verification
    gpu_supernodal_lu
    supernodal_gpu
)
if(TARGET gpu_ldlt_smoke)
    list(APPEND PROJECT_WARNING_TARGETS gpu_ldlt_smoke)
endif()
if(TARGET gpu_lu_smoke)
    list(APPEND PROJECT_WARNING_TARGETS gpu_lu_smoke)
endif()
if(TARGET unsymmetric_symbolic_smoke)
    list(APPEND PROJECT_WARNING_TARGETS unsymmetric_symbolic_smoke)
endif()
if(TARGET unsymmetric_matching_smoke)
    list(APPEND PROJECT_WARNING_TARGETS unsymmetric_matching_smoke)
endif()

if(MSVC)
    foreach(target_name IN LISTS PROJECT_WARNING_TARGETS)
        target_compile_options(${target_name} PRIVATE
            $<$<COMPILE_LANGUAGE:CXX>:/W4>
            $<$<COMPILE_LANGUAGE:CXX>:/utf-8>
            $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/W4>
            $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/utf-8>
        )
    endforeach()
else()
    foreach(target_name IN LISTS PROJECT_WARNING_TARGETS)
        target_compile_options(${target_name} PRIVATE
            $<$<COMPILE_LANGUAGE:CXX>:-Wall>
            $<$<COMPILE_LANGUAGE:CXX>:-Wextra>
            $<$<COMPILE_LANGUAGE:CXX>:-Wpedantic>
            $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall>
            $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wextra>
            $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wpedantic>
        )
    endforeach()
endif()
