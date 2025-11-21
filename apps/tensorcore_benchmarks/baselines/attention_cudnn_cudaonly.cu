#include <iostream>
#include <cmath>
#include <cublas_v2.h>
#include <cudnn.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <random>
#include <cublasLt.h>

#include "halide_benchmark.h"

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

inline void check_cudnn(cudnnStatus_t status, const char* msg = "", const char* file = __FILE__, int line = __LINE__) {
    if (status != CUDNN_STATUS_SUCCESS) {
        std::cerr << "cuDNN error: " << cudnnGetErrorString(status)
                  << " at " << file << ":" << line << " — " << msg << "\n";
        std::exit(EXIT_FAILURE);
    }
}

// === Globals & Helpers ===

#ifndef ATT_N
constexpr int N = 64;       // batch size
#else
constexpr int N = ATT_N;
#endif

#ifndef ATT_L
constexpr int L = 4096;     // sequence length
#else
constexpr int L = ATT_L;
#endif

#ifndef ATT_D
constexpr int D = 64;       // head dimension
#else
constexpr int D = ATT_D;
#endif

cublasHandle_t cublas_handle;
cublasLtHandle_t lt_handle;
cudnnHandle_t cudnn_handle;

half *d_Q, *d_K, *d_V, *d_probs_fp16;
float *d_scores, *d_probs, *d_output;

// First matmul (Q×K^T) descriptors
cublasLtMatrixLayout_t aDesc, bDesc, cDesc;
cublasLtMatmulDesc_t op;
cublasLtMatmulHeuristicResult_t heur;

// Second matmul (probs×V) descriptors
cublasLtMatrixLayout_t aDesc2, bDesc2, cDesc2;
cublasLtMatmulDesc_t op2;
cublasLtMatmulHeuristicResult_t heur2;

size_t workspaceSize = 1 << 30;
void* workspace = nullptr;

// === Step 1: Allocate & Initialize Buffers ===
inline half float_to_half(float x) {
    return __float2half(x);
}

void allocate_and_init() {
    // Set PEDANTIC math mode to disable tensor cores
    check_cublas(cublasSetMathMode(cublas_handle, CUBLAS_PEDANTIC_MATH), "setting pedantic math mode");

    size_t qkv_elems   = static_cast<size_t>(N) * L * D;
    size_t scores_elems = static_cast<size_t>(N) * L * L;
    size_t output_elems = static_cast<size_t>(N) * L * D;

    // === Allocate device memory ===
    check_cuda(cudaMalloc(&d_Q, qkv_elems * sizeof(half)), "allocating d_Q");
    check_cuda(cudaMalloc(&d_K, qkv_elems * sizeof(half)), "allocating d_K");
    check_cuda(cudaMalloc(&d_V, qkv_elems * sizeof(half)), "allocating d_V");

    check_cuda(cudaMalloc(&d_scores, scores_elems * sizeof(float)), "allocating d_scores");
    check_cuda(cudaMalloc(&d_probs,  scores_elems * sizeof(float)), "allocating d_probs");
    check_cuda(cudaMalloc(&d_probs_fp16, scores_elems * sizeof(half)), "allocating d_probs_fp16");
    check_cuda(cudaMalloc(&d_output, output_elems * sizeof(float)), "allocating d_output");

    // === Host-side initialization ===
    std::vector<half> h_Q(qkv_elems);
    std::vector<half> h_K(qkv_elems);
    std::vector<half> h_V(qkv_elems);

    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    for (size_t i = 0; i < qkv_elems; ++i) {
        h_Q[i] = __float2half(dist(gen));
        h_K[i] = __float2half(dist(gen));
        h_V[i] = __float2half(dist(gen));
    }

    check_cuda(cudaMemcpy(d_Q, h_Q.data(), qkv_elems * sizeof(half), cudaMemcpyHostToDevice),
               "copying Q to device");
    check_cuda(cudaMemcpy(d_K, h_K.data(), qkv_elems * sizeof(half), cudaMemcpyHostToDevice),
               "copying K to device");
    check_cuda(cudaMemcpy(d_V, h_V.data(), qkv_elems * sizeof(half), cudaMemcpyHostToDevice),
               "copying V to device");

    // scores = Q [LxD]  *  K^T [DxL]  -> [LxL]   (row-major everywhere)
    const int m = L, n = L, k = D;

     // 1) Operation descriptor (compute in FP32 without tensor cores)
    check_cublas(cublasLtMatmulDescCreate(&op, CUBLAS_COMPUTE_32F_PEDANTIC, CUDA_R_32F),
                 "create matmul desc");
    cublasOperation_t transA = CUBLAS_OP_N;  // Q
    cublasOperation_t transB = CUBLAS_OP_T;  // K^T
    check_cublas(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSA, &transA, sizeof(transA)),
                 "set transA");
    check_cublas(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSB, &transB, sizeof(transB)),
                 "set transB");

    // 2) Matrix layouts (ROW-MAJOR!)
    check_cublas(cublasLtMatrixLayoutCreate(&aDesc, CUDA_R_16F, /*rows*/m, /*cols*/k, /*ld*/k), "A layout");
    check_cublas(cublasLtMatrixLayoutCreate(&bDesc, CUDA_R_16F, /*rows*/n, /*cols*/k, /*ld*/k), "B layout");
    check_cublas(cublasLtMatrixLayoutCreate(&cDesc, CUDA_R_32F, /*rows*/m, /*cols*/n, /*ld*/n), "C layout");

    cublasLtOrder_t row = CUBLASLT_ORDER_ROW;
    check_cublas(cublasLtMatrixLayoutSetAttribute(aDesc, CUBLASLT_MATRIX_LAYOUT_ORDER, &row, sizeof(row)), "A order");
    check_cublas(cublasLtMatrixLayoutSetAttribute(bDesc, CUBLASLT_MATRIX_LAYOUT_ORDER, &row, sizeof(row)), "B order");
    check_cublas(cublasLtMatrixLayoutSetAttribute(cDesc, CUBLASLT_MATRIX_LAYOUT_ORDER, &row, sizeof(row)), "C order");

    // Strided batch in **elements** (not bytes)
    long long strideA = (long long)L * D;
    long long strideB = (long long)L * D;
    long long strideC = (long long)L * L;
    check_cublas(cublasLtMatrixLayoutSetAttribute(aDesc, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &N, sizeof(N)), "A bc");
    check_cublas(cublasLtMatrixLayoutSetAttribute(bDesc, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &N, sizeof(N)), "B bc");
    check_cublas(cublasLtMatrixLayoutSetAttribute(cDesc, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &N, sizeof(N)), "C bc");
    check_cublas(cublasLtMatrixLayoutSetAttribute(aDesc, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &strideA, sizeof(strideA)), "A stride");
    check_cublas(cublasLtMatrixLayoutSetAttribute(bDesc, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &strideB, sizeof(strideB)), "B stride");
    check_cublas(cublasLtMatrixLayoutSetAttribute(cDesc, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &strideC, sizeof(strideC)), "C stride");

    // 3) Heuristic pick
    cublasLtMatmulPreference_t pref;
    check_cublas(cublasLtMatmulPreferenceCreate(&pref), "pref");
    check_cuda(cudaMalloc(&workspace, workspaceSize), "alloc lt workspace");
    check_cublas(cublasLtMatmulPreferenceSetAttribute(
        pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspaceSize, sizeof(workspaceSize)), "pref ws");

    int returned = 0;
    cublasStatus_t st = cublasLtMatmulAlgoGetHeuristic(
        lt_handle, op, aDesc, bDesc, cDesc, cDesc, pref, 1, &heur, &returned);
    if (st != CUBLAS_STATUS_SUCCESS || returned == 0) {
        std::cerr << "Lt heuristic failed for QK^T (status=" << st << ", returned=" << returned << ")\n";
        std::exit(EXIT_FAILURE);
    }

    // === Setup second matmul: probs×V -> output ===
    // Dimensions: probs[LxL] × V[LxD] -> output[LxD]
    const int m2 = L, n2 = D, k2 = L;

    // Operation descriptor for probs×V (no tensor cores)
    check_cublas(cublasLtMatmulDescCreate(&op2, CUBLAS_COMPUTE_32F_PEDANTIC, CUDA_R_32F),
                 "create matmul desc 2");
    cublasOperation_t transA2 = CUBLAS_OP_N;  // probs
    cublasOperation_t transB2 = CUBLAS_OP_N;  // V
    check_cublas(cublasLtMatmulDescSetAttribute(op2, CUBLASLT_MATMUL_DESC_TRANSA, &transA2, sizeof(transA2)),
                 "set transA2");
    check_cublas(cublasLtMatmulDescSetAttribute(op2, CUBLASLT_MATMUL_DESC_TRANSB, &transB2, sizeof(transB2)),
                 "set transB2");

    // Matrix layouts for second matmul (ROW-MAJOR)
    check_cublas(cublasLtMatrixLayoutCreate(&aDesc2, CUDA_R_16F, /*rows*/m2, /*cols*/k2, /*ld*/k2), "A2 layout");
    check_cublas(cublasLtMatrixLayoutCreate(&bDesc2, CUDA_R_16F, /*rows*/k2, /*cols*/n2, /*ld*/n2), "B2 layout");
    check_cublas(cublasLtMatrixLayoutCreate(&cDesc2, CUDA_R_32F, /*rows*/m2, /*cols*/n2, /*ld*/n2), "C2 layout");

    check_cublas(cublasLtMatrixLayoutSetAttribute(aDesc2, CUBLASLT_MATRIX_LAYOUT_ORDER, &row, sizeof(row)), "A2 order");
    check_cublas(cublasLtMatrixLayoutSetAttribute(bDesc2, CUBLASLT_MATRIX_LAYOUT_ORDER, &row, sizeof(row)), "B2 order");
    check_cublas(cublasLtMatrixLayoutSetAttribute(cDesc2, CUBLASLT_MATRIX_LAYOUT_ORDER, &row, sizeof(row)), "C2 order");

    // Strided batch for second matmul
    long long strideA2 = (long long)L * L;  // probs
    long long strideB2 = (long long)L * D;  // V
    long long strideC2 = (long long)L * D;  // output
    check_cublas(cublasLtMatrixLayoutSetAttribute(aDesc2, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &N, sizeof(N)), "A2 bc");
    check_cublas(cublasLtMatrixLayoutSetAttribute(bDesc2, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &N, sizeof(N)), "B2 bc");
    check_cublas(cublasLtMatrixLayoutSetAttribute(cDesc2, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &N, sizeof(N)), "C2 bc");
    check_cublas(cublasLtMatrixLayoutSetAttribute(aDesc2, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &strideA2, sizeof(strideA2)), "A2 stride");
    check_cublas(cublasLtMatrixLayoutSetAttribute(bDesc2, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &strideB2, sizeof(strideB2)), "B2 stride");
    check_cublas(cublasLtMatrixLayoutSetAttribute(cDesc2, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &strideC2, sizeof(strideC2)), "C2 stride");

    // Heuristic for second matmul
    int returned2 = 0;
    cublasStatus_t st2 = cublasLtMatmulAlgoGetHeuristic(
        lt_handle, op2, aDesc2, bDesc2, cDesc2, cDesc2, pref, 1, &heur2, &returned2);
    if (st2 != CUBLAS_STATUS_SUCCESS || returned2 == 0) {
        std::cerr << "Lt heuristic failed for probs×V (status=" << st2 << ", returned=" << returned2 << ")\n";
        std::exit(EXIT_FAILURE);
    }

    cublasLtMatmulPreferenceDestroy(pref);
}

// === Step 2: Compute Scores = (Q x K^T) / sqrt(D) ===
__global__ void scale_kernel(float* data, size_t numel, float scale) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) {
        data[idx] *= scale;
    }
}

void scale_scores() {
    size_t numel = static_cast<size_t>(N) * L * L;
    float scale = 1.0f / std::sqrt(static_cast<float>(D));

    int threads = 256;
    int blocks = static_cast<int>((numel + threads - 1) / threads);
    scale_kernel<<<blocks, threads>>>(d_scores, numel, scale);
    check_cuda(cudaGetLastError(), "launching scale kernel");
}

void compute_scores() {
    const float alpha = 1.0f, beta = 0.0f;

    check_cublas(cublasLtMatmul(lt_handle, op,
                                &alpha,
                                d_Q, aDesc,
                                d_K, bDesc,
                                &beta,
                                d_scores, cDesc,
                                d_scores, cDesc,
                                &heur.algo,
                                workspace, workspaceSize, 0),
                 "lt matmul QK^T");

    scale_scores();
}

// === Step 3: Softmax using cuDNN ===
__global__ void f32_to_f16_kernel(const float* __restrict__ in, half* __restrict__ out, size_t n) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = __float2half(in[i]);
}

void cast_probs_to_half() {
    size_t numel = static_cast<size_t>(N) * L * L;
    int threads = 256;
    int blocks = static_cast<int>((numel + threads - 1) / threads);
    f32_to_f16_kernel<<<blocks, threads>>>(d_probs, d_probs_fp16, numel);
    check_cuda(cudaGetLastError(), "casting probs float->half");
}

void softmax_scores() {
    cudnnTensorDescriptor_t desc;
    check_cudnn(cudnnCreateTensorDescriptor(&desc), "create tensor desc");

    // Treat as (N*L) batches of length L vectors
    int n_vecs = N * L;
    check_cudnn(
        cudnnSetTensor4dDescriptor(
            desc,
            CUDNN_TENSOR_NCHW,
            CUDNN_DATA_FLOAT,
            n_vecs,   // N = number of rows
            L,        // C = number of columns -> normalize across this
            1, 1),    // H=W=1
        "set tensor desc");

    float alpha = 1.0f, beta = 0.0f;
    check_cudnn(
        cudnnSoftmaxForward(
            cudnn_handle,
            CUDNN_SOFTMAX_ACCURATE,
            CUDNN_SOFTMAX_MODE_CHANNEL,
            &alpha,
            desc, d_scores,
            &beta,
            desc, d_probs),
        "cudnnSoftmaxForward");

    cast_probs_to_half();

    check_cudnn(cudnnDestroyTensorDescriptor(desc), "destroy tensor desc");
}

// === Step 4: Compute Output = probs x V ===
void compute_output() {
    const float alpha = 1.0f, beta = 0.0f;

    check_cublas(cublasLtMatmul(lt_handle, op2,
                                &alpha,
                                d_probs_fp16, aDesc2,
                                d_V, bDesc2,
                                &beta,
                                d_output, cDesc2,
                                d_output, cDesc2,
                                &heur2.algo,
                                workspace, workspaceSize, 0),
                 "lt matmul probs×V");
}

// === Step 5: Benchmark Harness ===
void run_attention_benchmark() {
    std::cout << "Running attention with:" << std::endl;
    std::cout << "  Input size: N=" << N << " L=" << L << " D=" << D << std::endl;
    std::cout << "  Schedule: cudnn_cudaonly (no tensor cores)" << std::endl;

    allocate_and_init();

    auto time = Halide::Tools::benchmark(5, 5, [&]() {
        compute_scores();
        softmax_scores();
        compute_output();
        cudaDeviceSynchronize();
    });

    std::cout << "Runtime: " << time << "\n";
}

// === Main ===
int main() {
    check_cublas(cublasCreate(&cublas_handle), "create cublas handle");
    check_cudnn(cudnnCreate(&cudnn_handle), "create cudnn handle");
    check_cublas(cublasLtCreate(&lt_handle), "create cublasLt handle");

    run_attention_benchmark();

    cudaFree(workspace);
    cublasLtMatrixLayoutDestroy(aDesc);
    cublasLtMatrixLayoutDestroy(bDesc);
    cublasLtMatrixLayoutDestroy(cDesc);
    cublasLtMatmulDescDestroy(op);
    cublasLtMatrixLayoutDestroy(aDesc2);
    cublasLtMatrixLayoutDestroy(bDesc2);
    cublasLtMatrixLayoutDestroy(cDesc2);
    cublasLtMatmulDescDestroy(op2);

    cudaFree(d_Q);
    cudaFree(d_K);
    cudaFree(d_V);
    cudaFree(d_scores);
    cudaFree(d_probs);
    cudaFree(d_probs_fp16);
    cudaFree(d_output);
    cublasDestroy(cublas_handle);
    cudnnDestroy(cudnn_handle);
    cublasLtDestroy(lt_handle);
    return 0;
}
