#include "HalideBuffer.h"
#include "halide_benchmark.h"
                                                                      
#include <iostream>
#include <cstdlib>  // for rand()
#include <iomanip>  // for std::fixed and std::setprecision                                                       
                                                                      
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

inline void check_cuda(cudaError_t err, const char* msg = "", const char* file = __FILE__, int line = __LINE__) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(err)
                  << " at " << file << ":" << line << " — " << msg << "\n";
        std::exit(EXIT_FAILURE);
    }
}

inline void check_cublas(cublasStatus_t status, const char* msg = "", const char* file = __FILE__, int line = __LINE__) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        std::cerr << "cuBLAS error at " << file << ":" << line << " — " << msg << "\n";
        std::exit(EXIT_FAILURE);
    }
}
                                                                      
using namespace Halide::Runtime;                                      
using namespace Halide::Tools; 

half *d_A, *d_B;
float *d_C;

void matmul_cublas(const half* h_A, const half* h_B, float* h_C, int M, int N, int K) {
    cublasHandle_t handle;
    check_cublas(cublasCreate(&handle), "creating cuBLAS handle");

    // Allocate device memory
    check_cuda(cudaMalloc(&d_A, M * K * sizeof(half)), "allocating d_A");
    check_cuda(cudaMalloc(&d_B, K * N * sizeof(half)), "allocating d_B");
    check_cuda(cudaMalloc(&d_C, M * N * sizeof(float)), "allocating d_C");

    check_cuda(cudaMemcpy(d_A, h_A, M * K * sizeof(half), cudaMemcpyHostToDevice), "copying h_A");
    check_cuda(cudaMemcpy(d_B, h_B, K * N * sizeof(half), cudaMemcpyHostToDevice), "copying h_B");

    float alpha = 1.0f;
    float beta = 0.0f;

    check_cublas(cublasGemmEx(
        handle,
        CUBLAS_OP_N,
        CUBLAS_OP_N,
        N, M, K,
        &alpha,
        d_B, CUDA_R_16F, N,
        d_A, CUDA_R_16F, K,
        &beta,
        d_C, CUDA_R_32F, N,
        CUBLAS_COMPUTE_32F_FAST_16F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP), "running cublasGemmEx");
}

int main(int argc, char **argv) {                                     
    // Create test data using compile-time definitions                
    const int M = 4096;//MATMUL_M;                                    
    const int N = 4096;//MATMUL_N;                                         
    const int K = 4096;//MATMUL_K;
    
    std::string benchmark_name = "matmul";//BENCHMARK_NAME;

    std::cout << "Running " << benchmark_name << " with:" << std::endl;
    std::cout << "  Matrix size: " << M << "x" << N << "x" << K << std::endl;
    std::cout << "  Schedule: cublas" << std::endl;

    // Create matrix buffers with random values
    half* h_A = new half[M * K];
    half* h_B = new half[K * N];
    float* h_C = new float[M * N];

    for (int i = 0; i < M * K; ++i) h_A[i] = __float2half(1.0f);
    for (int i = 0; i < K * N; ++i) h_B[i] = __float2half(1.0f);

    cublasHandle_t handle;
    check_cublas(cublasCreate(&handle), "creating cuBLAS handle");

    // Call the generated function
    auto time = benchmark(5, 5, [&]() {   
        matmul_cublas(h_A, h_B, h_C, M, N, K);
        check_cuda(cudaDeviceSynchronize(), "sync after matmul_cublas");
    });

    check_cuda(cudaMemcpy(h_C, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost), "copying d_C to host");

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    std::cout << "Runtime: " << time << "\n";

    return 0;
}