#pragma once

#include <cuda_runtime.h>
#include <stdexcept>


class CudaException : public std::exception {
private:
    cudaError_t res;

public:
    CudaException(cudaError_t res) : res(res) { }

    static void check(cudaError_t res)
    {
        if (res != cudaSuccess) {
            throw CudaException(res);
        }
    }
};
