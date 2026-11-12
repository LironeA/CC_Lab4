cmake_minimum_required(VERSION 3.18)
project(lab4_cuda_video_difference LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(OpenCV REQUIRED COMPONENTS core videoio cudaarithm)
find_package(Threads REQUIRED)

add_executable(cuda_video_difference src/cuda_video_difference.cpp)
target_link_libraries(cuda_video_difference PRIVATE ${OpenCV_LIBS} Threads::Threads)
target_compile_options(cuda_video_difference PRIVATE
    $<$<CXX_COMPILER_ID:MSVC>:/W4>
    $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-Wall -Wextra -Wpedantic>
)
