#pragma once

#include <cstddef>

int available_gpu_count();
void difference_on_gpu(int device, const unsigned char* previous, const unsigned char* current,
                       unsigned char* difference, std::size_t bytes);
