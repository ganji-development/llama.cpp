if (NOT EXISTS $ENV{ROCM_PATH})
    if (NOT EXISTS /opt/rocm)
        set(ROCM_PATH /usr)
    else()
        set(ROCM_PATH /opt/rocm)
    endif()
else()
    set(ROCM_PATH $ENV{ROCM_PATH})
endif()

list(APPEND CMAKE_PREFIX_PATH  ${ROCM_PATH})
list(APPEND CMAKE_PREFIX_PATH "${ROCM_PATH}/lib64/cmake")

if (NOT DEFINED CMAKE_HIP_FLAGS_DEBUG)
    set(CMAKE_HIP_FLAGS_DEBUG "-g -O2")
endif()

# CMake on Windows doesn't support the HIP language yet
if (WIN32)
    set(CXX_IS_HIPCC TRUE)
else()
    string(REGEX MATCH "hipcc(\.bat)?$" CXX_IS_HIPCC "${CMAKE_CXX_COMPILER}")
endif()

if (CXX_IS_HIPCC)
    if (LINUX)
        if (NOT ${CMAKE_CXX_COMPILER_ID} MATCHES "Clang")
            message(WARNING "Only LLVM is supported for HIP, hint: CXX=/opt/rocm/llvm/bin/clang++")
        endif()

        message(WARNING "Setting hipcc as the C++ compiler is legacy behavior."
                " Prefer setting the HIP compiler directly. See README for details.")
    endif()
else()
    # Forward (AMD)GPU_TARGETS to CMAKE_HIP_ARCHITECTURES.
    if(AMDGPU_TARGETS AND NOT GPU_TARGETS)
        set(GPU_TARGETS ${AMDGPU_TARGETS})
    endif()
    if(GPU_TARGETS AND NOT CMAKE_HIP_ARCHITECTURES)
        set(CMAKE_HIP_ARCHITECTURES ${GPU_TARGETS})
    endif()
    cmake_minimum_required(VERSION 3.21)
    enable_language(HIP)
endif()

find_package(hip     REQUIRED)
find_package(hipblas REQUIRED)
find_package(rocblas REQUIRED)

if (GGML_HIP_RCCL)
    find_package(rccl REQUIRED)
endif()

if (${hip_VERSION} VERSION_LESS 6.1)
    message(FATAL_ERROR "At least ROCM/HIP V6.1 is required")
endif()

message(STATUS "HIP and hipBLAS found")

file(GLOB   GGML_HEADERS_ROCM "../ggml-cuda/*.cuh")
list(APPEND GGML_HEADERS_ROCM "../../include/ggml-cuda.h")

file(GLOB   GGML_SOURCES_ROCM "../ggml-cuda/*.cu")
file(GLOB   SRCS "../ggml-cuda/template-instances/fattn-tile*.cu")
list(APPEND GGML_SOURCES_ROCM ${SRCS})
file(GLOB   SRCS "../ggml-cuda/template-instances/fattn-mma*.cu")
list(APPEND GGML_SOURCES_ROCM ${SRCS})
file(GLOB   SRCS "../ggml-cuda/template-instances/mmq*.cu")
list(APPEND GGML_SOURCES_ROCM ${SRCS})
file(GLOB   SRCS "../ggml-cuda/template-instances/mmf*.cu")
list(APPEND GGML_SOURCES_ROCM ${SRCS})

if (GGML_CUDA_FA_ALL_QUANTS)
    file(GLOB   SRCS "../ggml-cuda/template-instances/fattn-vec*.cu")
    list(APPEND GGML_SOURCES_ROCM ${SRCS})
    add_compile_definitions(GGML_CUDA_FA_ALL_QUANTS)
else()
    list(APPEND GGML_SOURCES_ROCM
        ../ggml-cuda/template-instances/fattn-vec-instance-f16-f16.cu
        ../ggml-cuda/template-instances/fattn-vec-instance-q4_0-q4_0.cu
        ../ggml-cuda/template-instances/fattn-vec-instance-q8_0-q8_0.cu
        ../ggml-cuda/template-instances/fattn-vec-instance-bf16-bf16.cu)
endif()

ggml_add_backend_library(ggml-hip
                         ${GGML_HEADERS_ROCM}
                         ${GGML_SOURCES_ROCM}
                        )

# TODO: do not use CUDA definitions for HIP
if (NOT GGML_BACKEND_DL)
    target_compile_definitions(ggml PUBLIC GGML_USE_CUDA)
endif()

add_compile_definitions(GGML_USE_HIP)

if (GGML_CUDA_FORCE_MMQ)
    add_compile_definitions(GGML_CUDA_FORCE_MMQ)
endif()

if (GGML_CUDA_FORCE_CUBLAS)
    add_compile_definitions(GGML_CUDA_FORCE_CUBLAS)
endif()

if (GGML_CUDA_NO_PEER_COPY)
    add_compile_definitions(GGML_CUDA_NO_PEER_COPY)
endif()

if (GGML_HIP_GRAPHS)
    add_compile_definitions(GGML_HIP_GRAPHS)
endif()

if (GGML_HIP_NO_VMM)
    add_compile_definitions(GGML_HIP_NO_VMM)
endif()

if (NOT GGML_HIP_MMQ_MFMA)
    add_compile_definitions(GGML_HIP_NO_MMQ_MFMA)
endif()

if (GGML_HIP_RCCL)
    add_compile_definitions(GGML_USE_NCCL) # RCCL has the same interface as NCCL.
endif()

if (GGML_HIP_EXPORT_METRICS)
    set(CMAKE_HIP_FLAGS "${CMAKE_HIP_FLAGS} -Rpass-analysis=kernel-resource-usage --save-temps")
endif()

# Fast math for HIP, like CUDA's -use_fast_math. Not -ffast-math: that implies -ffinite-math-only, which breaks ggml's INFINITY masking and produces NaNs.
set(CMAKE_HIP_FLAGS "${CMAKE_HIP_FLAGS} -funsafe-math-optimizations")

if (NOT GGML_CUDA_FA)
    add_compile_definitions(GGML_CUDA_NO_FA)
endif()

if (CXX_IS_HIPCC)
    set_source_files_properties(${GGML_SOURCES_ROCM} PROPERTIES LANGUAGE CXX)
    if (WIN32 AND CMAKE_BUILD_TYPE STREQUAL "Debug")
        # CMake on Windows doesn't support the HIP language yet.
        # Therefore we workaround debug build's failure on HIP backend this way.
        set_source_files_properties(${GGML_SOURCES_ROCM} PROPERTIES COMPILE_FLAGS "-O2 -g")
    endif()
    target_link_libraries(ggml-hip PRIVATE hip::device)
else()
    set_source_files_properties(${GGML_SOURCES_ROCM} PROPERTIES LANGUAGE HIP)
endif()

if (GGML_STATIC)
    message(FATAL_ERROR "Static linking not supported for HIP/ROCm")
endif()

if (GGML_HIP_RCCL)
    target_link_libraries(ggml-hip PRIVATE ggml-base roc::rccl)
endif()

target_link_libraries(ggml-hip PRIVATE ggml-base hip::host roc::rocblas roc::hipblas)

# Stage the HIP runtime beside the built binaries (Windows).
#
# The HIP runtime loaded at run time must be the one this backend was compiled
# against. With none present next to the binaries, Windows resolves amdhip64
# from System32 - the display driver's copy - because the dependency comes in
# through ggml-hip's import table, which PATH order does not affect. If that
# copy's version differs from the SDK used here, the mismatch does not fail
# cleanly at load: hipMemGetInfo starts returning "invalid argument" and the
# process later faults inside the HIP runtime on one of its own worker threads.
if (WIN32)
    set(GGML_HIP_RUNTIME_DIRS "")
    if (DEFINED ENV{HIP_PATH})
        list(APPEND GGML_HIP_RUNTIME_DIRS "$ENV{HIP_PATH}/bin")
    endif()
    list(APPEND GGML_HIP_RUNTIME_DIRS "${ROCM_PATH}/bin")

    set(GGML_HIP_RUNTIME_DLLS "")
    foreach (dir IN LISTS GGML_HIP_RUNTIME_DIRS)
        if (EXISTS "${dir}")
            # rocsolver, rocm_kpack and origami are not linked here; they come
            # in transitively through hipblas, rocblas and hipblaslt, so they
            # have to be staged too or the executables fail to start with
            # 0xC0000135. Which of them exist varies by ROCm build, hence the
            # glob: absent ones simply contribute nothing.
            file(GLOB found
                "${dir}/amdhip64*.dll"
                "${dir}/amd_comgr*.dll"
                "${dir}/hipblas.dll"
                "${dir}/libhipblaslt.dll"
                "${dir}/rocblas.dll"
                "${dir}/rocsolver.dll"
                "${dir}/rocm_kpack.dll"
                "${dir}/origami.dll")
            list(APPEND GGML_HIP_RUNTIME_DLLS ${found})
        endif()
    endforeach()

    if (GGML_HIP_RUNTIME_DLLS)
        list(REMOVE_DUPLICATES GGML_HIP_RUNTIME_DLLS)
        foreach (dll IN LISTS GGML_HIP_RUNTIME_DLLS)
            add_custom_command(TARGET ggml-hip POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                        "${dll}" "$<TARGET_FILE_DIR:ggml-hip>")
        endforeach()
        message(STATUS "ggml-hip: staging HIP runtime from ${GGML_HIP_RUNTIME_DIRS}")

        # rocblas.dll resolves its Tensile kernels relative to the loading
        # executable, not to itself, so the data has to sit beside every binary
        # or rocBLAS aborts the process on first gemm with 0xC0000409.
        foreach (dir IN LISTS GGML_HIP_RUNTIME_DIRS)
            if (EXISTS "${dir}/rocblas/library")
                add_custom_command(TARGET ggml-hip POST_BUILD
                    COMMAND ${CMAKE_COMMAND} -E copy_directory
                            "${dir}/rocblas/library"
                            "$<TARGET_FILE_DIR:ggml-hip>/rocblas/library")
                break()
            endif()
        endforeach()
    else()
        message(WARNING
            "ggml-hip: no HIP runtime DLLs found to stage; the process will bind "
            "the driver's copy from System32, which may not match this SDK. "
            "Set HIP_PATH or ROCM_PATH to the SDK you built against.")
    endif()
endif()
