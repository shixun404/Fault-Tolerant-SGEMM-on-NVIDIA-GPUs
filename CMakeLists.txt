cmake_minimum_required(VERSION 3.18)
project(ft_sgemm LANGUAGES CXX CUDA)


set(CMAKE_CUDA_ARCHITECTURES 80)


set(CMAKE_CUDA_STANDARD 11)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3")

find_package(CUDAToolkit REQUIRED)

# 包含头文件路径
include_directories(
    ${CUDAToolkit_INCLUDE_DIRS}
    cuda-samples/Common/
    ${CMAKE_SOURCE_DIR}
)


set(KERNEL_PATH "kernel/ft_sgemm")


set(SRC_FILES
    ${KERNEL_PATH}/sgemm.cu
    utils/utils.cu
)


add_executable(ft_sgemm ${SRC_FILES})


target_link_libraries(ft_sgemm PRIVATE CUDA::cudart CUDA::cublas)
