#pragma once

#include <cuda_runtime.h>
#include <stdexcept>


class CudaException : public std::exception {
private:
    cudaError_t res;
    std::string message;

public:
    CudaException(cudaError_t res) 
        : res(res), message(std::string(cudaGetErrorString(res))) {}

    const char* what() const noexcept override {
        return message.c_str();
    }
    static void check(cudaError_t res)
    {
        if (res != cudaSuccess) {
            throw CudaException(res);
        }
    }
};
