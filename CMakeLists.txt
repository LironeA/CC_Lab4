cmake_minimum_required(VERSION 3.18)
project(lab4_cuda_video_difference LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED COMPONENTS core videoio)
find_package(Threads REQUIRED)
find_package(CUDAToolkit REQUIRED)

add_executable(cuda_video_difference
    src/cuda_video_difference.cpp
    src/frame_difference.cu
)
target_link_libraries(cuda_video_difference PRIVATE ${OpenCV_LIBS} Threads::Threads CUDA::cudart)
target_compile_options(cuda_video_difference PRIVATE
    $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CXX_COMPILER_ID:MSVC>>:/W4>
    $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<NOT:$<CXX_COMPILER_ID:MSVC>>>:-Wall -Wextra -Wpedantic>
)
