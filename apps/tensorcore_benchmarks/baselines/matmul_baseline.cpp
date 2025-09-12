#include "halide_benchmark.h"
                                                                      
#include <iostream>
#include <cstdlib>  // for rand()
#include <iomanip>  // for std::fixed and std::setprecision                                                       
                                                                      
#include <cublas_v2.h>
#include <cublasLt.h>
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
                                                                                             
using namespace Halide::Tools; 

cublasHandle_t handle;
cublasLtHandle_t lt_handle;

half *d_A, *d_B;
float *d_C;

cublasLtMatmulAlgo_t algo;
cublasLtMatmulDesc_t op_desc;
cublasLtMatrixLayout_t a_desc, b_desc, c_desc;

cublasLtMatmulPreference_t preference;

void* workspace = nullptr;
size_t workspace_size = 1 << 24; // 16 MB

void matmul_cublasLt(const half* h_A, const half* h_B, float* h_C, int M, int N, int K) {
    float alpha = 1.0f;
    float beta = 0.0f;

    // Launch
    check_cublas(cublasLtMatmul(
        lt_handle,
        op_desc,
        &alpha,
        d_A, a_desc,
        d_B, b_desc,
        &beta,
        d_C, c_desc,
        d_C, c_desc,
        &algo,
        workspace,
        workspace_size,
        0), "running LtMatmul");
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

    // Create cublas handles
    check_cublas(cublasCreate(&handle), "creating cuBLAS handle");
    check_cublas(cublasLtCreate(&lt_handle), "creating cuBLASLt handle");

    // Describe matrices
    check_cublas(cublasLtMatrixLayoutCreate(&a_desc, CUDA_R_16F, K, M, K), "creating A desc");
    check_cublas(cublasLtMatrixLayoutCreate(&b_desc, CUDA_R_16F, N, K, N), "creating B desc");
    check_cublas(cublasLtMatrixLayoutCreate(&c_desc, CUDA_R_32F, N, M, N), "creating C desc");

    // Operation descriptor
    check_cublas(cublasLtMatmulDescCreate(&op_desc, CUBLAS_COMPUTE_32F_FAST_16F, CUDA_R_32F), "creating op desc");
    
    // After creating the layouts, set them to column-major
    cublasLtOrder_t order = CUBLASLT_ORDER_COL;
    check_cublas(cublasLtMatrixLayoutSetAttribute(a_desc, CUBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order)), "setting A order");
    check_cublas(cublasLtMatrixLayoutSetAttribute(b_desc, CUBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order)), "setting B order");
    check_cublas(cublasLtMatrixLayoutSetAttribute(c_desc, CUBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order)), "setting C order");

    // Workspace
    check_cuda(cudaMalloc(&workspace, workspace_size), "allocating workspace");

    
    check_cublas(cublasLtMatmulPreferenceCreate(&preference), "creating preference");
    check_cublas(cublasLtMatmulPreferenceSetAttribute(
        preference,
        CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
        &workspace_size,
        sizeof(workspace_size)), "setting workspace size");
        

    // Heuristic search (picks the best kernel config)
    cublasLtMatmulHeuristicResult_t heuristicResult;
    int returnedResults = 0;

    check_cublas(cublasLtMatmulAlgoGetHeuristic(
        lt_handle,
        op_desc,
        a_desc,
        b_desc,
        c_desc,
        c_desc,
        preference,
        1,                      // requestedAlgoCount
        &heuristicResult,       // heuristicResultsArray
        &returnedResults),      // returnAlgoCount
        "getting heuristic"
    );

    algo = heuristicResult.algo;

    // Allocate device memory
    check_cuda(cudaMalloc(&d_A, M * K * sizeof(half)), "allocating d_A");
    check_cuda(cudaMalloc(&d_B, K * N * sizeof(half)), "allocating d_B");
    check_cuda(cudaMalloc(&d_C, M * N * sizeof(float)), "allocating d_C");

    check_cuda(cudaMemcpy(d_A, h_A, M * K * sizeof(half), cudaMemcpyHostToDevice), "copying h_A");
    check_cuda(cudaMemcpy(d_B, h_B, K * N * sizeof(half), cudaMemcpyHostToDevice), "copying h_B");

    // Call the generated function
    auto time = benchmark(5, 5, [&]() {   
        matmul_cublasLt(h_A, h_B, h_C, M, N, K);
        check_cuda(cudaDeviceSynchronize(), "sync after matmul_cublas");
    });

    check_cuda(cudaMemcpy(h_C, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost), "copying d_C to host");

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    // Cleanup
    cudaFree(workspace);
    cublasLtMatrixLayoutDestroy(a_desc);
    cublasLtMatrixLayoutDestroy(b_desc);
    cublasLtMatrixLayoutDestroy(c_desc);
    cublasLtMatmulDescDestroy(op_desc);
    cublasLtDestroy(lt_handle);

    std::cout << "Runtime: " << std::fixed << std::setprecision(9) << time << "\n";

    return 0;
}