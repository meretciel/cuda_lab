cmake_minimum_required(VERSION 3.30)
project(cuda_lab LANGUAGES CUDA CXX C)

set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS  OFF)
set(CMAKE_CUDA_FLAGS "-Wall")

add_executable(hello_world src/hello_world.cu)
set_target_properties(hello_world PROPERTIES
        CUDA_STANDARD 17
        CUDA_STANDARD_REQUIRED ON
)

add_executable(sum_vector src/sum_vector.cu)
set_target_properties(sum_vector PROPERTIES
        CUDA_STANDARD 17
        CUDA_STANDARD_REQUIRED ON
)


add_executable(sum_vector_v2 src/sum_vector_v2.cu)
set_target_properties(sum_vector_v2 PROPERTIES
        CUDA_STANDARD 17
        CUDA_STANDARD_REQUIRED ON
)


# Matrix Multiplication
add_executable(matrix_multiplication src/matrix_multiplication.cu)
set_target_properties(matrix_multiplication PROPERTIES
        CUDA_STANDARD 17
        CUDA_STANDARD_REQUIRED ON
)


add_executable(print_cuda_properties src/print_cuda_properties.cu)
set_target_properties(print_cuda_properties PROPERTIES
        CUDA_STANDARD 17
        CUDA_STANDARD_REQUIRED ON
)
