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

constexpr int N = 64;       // batch size
constexpr int L = 4096;     // sequence length
constexpr int D = 64;       // head dimension

cublasHandle_t cublas_handle;
cublasLtHandle_t lt_handle;
cudnnHandle_t cudnn_handle;

half *d_Q, *d_K, *d_V, *d_probs_fp16;
float *d_scores, *d_probs, *d_output;

cublasLtMatrixLayout_t aDesc, bDesc, cDesc;
cublasLtMatmulDesc_t op;
size_t workspaceSize = 1 << 30;
void* workspace = nullptr;
cublasLtMatmulHeuristicResult_t heur;

// TODO: Add error-check macros for cuDNN similar to check_cublas/check_cuda

// === Step 1: Allocate & Initialize Buffers ===
// Utility: convert float -> half safely
inline half float_to_half(float x) {
    return __float2half(x);
}

void allocate_and_init() {
    cublasMath_t mathMode = CUBLAS_TENSOR_OP_MATH;
    cublasSetMathMode(cublas_handle, mathMode);

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

    // === Sanity check: print first few Q values ===
    /*std::vector<half> h_Q_check(8);
    check_cuda(cudaMemcpy(h_Q_check.data(), d_Q, 8 * sizeof(half), cudaMemcpyDeviceToHost),
               "copying Q back to host");
    std::cout << "First few Q values (as float): ";
    for (int i = 0; i < 8; ++i) {
        std::cout << __half2float(h_Q_check[i]) << " ";
    }
    std::cout << "\n";*/

    // scores = Q [LxD]  *  K^T [DxL]  -> [LxL]   (row-major everywhere)
    const int m = L, n = L, k = D;

     // 1) Operation descriptor (compute in FP32, inputs FP16, output FP32)
    check_cublas(cublasLtMatmulDescCreate(&op, CUBLAS_COMPUTE_32F_FAST_16F, CUDA_R_32F),
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
        std::cerr << "Lt heuristic failed (status=" << st << ", returned=" << returned << ")\n";
        std::exit(EXIT_FAILURE);
    }
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

void compute_scores_old() {
    // Dimensions per batch
    int m = L; // rows of Q
    int n = L; // cols of K^T
    int k = D; // inner dimension

    // Leading dimensions (row-major trick):
    // - Treat row-major Q (LxD) as column-major Q^T (D x L), so we mark opA = CUBLAS_OP_T
    // - Similarly treat row-major K (LxD) as column-major K^T (D x L) and also transpose to get K^T^T = K
    //   Actually, we want K^T, so we can just use opB = CUBLAS_OP_N (since column-major layout is already transposed).
    cublasOperation_t opA = CUBLAS_OP_N;
    cublasOperation_t opB = CUBLAS_OP_T;

    // Strides between consecutive batches
    long long strideA = static_cast<long long>(L) * D;  // Q
    long long strideB = static_cast<long long>(L) * D;  // K
    long long strideC = static_cast<long long>(L) * L;  // scores

    float alpha = 1.0f;
    float beta  = 0.0f;

    check_cublas(cublasGemmStridedBatchedEx(
        cublas_handle,
        opB, opA,         // NOTE: cublas uses column-major, so order reversed: C = B^T * A^T
        n, m, k,          // n = cols of C, m = rows of C, k = inner dimension
        &alpha,
        d_K, CUDA_R_16F, k, strideB,  // B matrix
        d_Q, CUDA_R_16F, k, strideA,  // A matrix
        &beta,
        d_scores, CUDA_R_32F, n, strideC, // C matrix
        N,
        CUBLAS_COMPUTE_32F_FAST_16F,
        CUBLAS_GEMM_DEFAULT));

    scale_scores();

    // Optional: sanity-check a few values
    /*std::vector<float> h_scores(8);
    check_cuda(cudaMemcpy(h_scores.data(), d_scores, 8 * sizeof(float), cudaMemcpyDeviceToHost),
               "copying scores back to host");
    std::cout << "First few scores: ";
    for (float val : h_scores) {
        std::cout << val << " ";
    }
    std::cout << "\n";*/
}

void compute_scores() {
    const float alpha = 1.0f, beta = 0.0f;

    // 4) Execute
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

    scale_scores();  // keep your scaling kernel
}

// === Step 3: Softmax using cuDNN ===

// add after softmax_scores()
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
            CUDNN_SOFTMAX_ACCURATE,   // or FAST for perf
            CUDNN_SOFTMAX_MODE_CHANNEL,  // normalize across C dim
            &alpha,
            desc, d_scores,
            &beta,
            desc, d_probs),
        "cudnnSoftmaxForward");

    cast_probs_to_half();

    // Destroy descriptor to avoid leaks
    check_cudnn(cudnnDestroyTensorDescriptor(desc), "destroy tensor desc");

    // === Sanity check: copy first row of first batch to host and print ===
    /*std::vector<float> h_probs(L);
    check_cuda(cudaMemcpy(h_probs.data(), d_probs, L * sizeof(float), cudaMemcpyDeviceToHost),
               "copying probs back to host");
    std::cout << "First row softmax probs: ";
    for (int i = 0; i < std::min(L, 8); ++i) {
        std::cout << h_probs[i] << " ";
    }
    std::cout << "... (sum ≈ " 
              << std::accumulate(h_probs.begin(), h_probs.end(), 0.0f)
              << ")\n";*/
}

// === Step 4: Compute Output = probs x V ===
void compute_output() {
    int m = L;  // rows of A (probs)
    int n = D;  // cols of B (V)
    int k = L;  // shared dim

    cublasOperation_t opA = CUBLAS_OP_T;
    cublasOperation_t opB = CUBLAS_OP_T;

    int lda = L; // cols of A in row-major
    int ldb = L; // cols of B in row-major
    int ldc = D; // cols of C in row-major

    long long strideA = static_cast<long long>(L) * L;  // probs
    long long strideB = static_cast<long long>(L) * D;  // V
    long long strideC = static_cast<long long>(L) * D;  // output

    float alpha = 1.0f;
    float beta  = 0.0f;

    cublasStatus_t stat = cublasGemmStridedBatchedEx(
        cublas_handle,
        opB, opA,
        n, m, k,
        &alpha,
        d_V,          CUDA_R_16F, ldb, strideB,   // B
        d_probs_fp16, CUDA_R_16F, lda, strideA,   // A (now half)
        &beta,
        d_output,     CUDA_R_32F, ldc, strideC,   // C
        N,
        CUBLAS_COMPUTE_32F_FAST_16F,
        CUBLAS_GEMM_DEFAULT);

    if (stat != CUBLAS_STATUS_SUCCESS) {
        std::cerr << "cuBLAS error in compute_output: " << stat << "\n";
        std::exit(EXIT_FAILURE);
    }

    /*std::vector<float> h_out(8);
    check_cuda(cudaMemcpy(h_out.data(), d_output, 8 * sizeof(float), cudaMemcpyDeviceToHost),
               "copying output back to host");
    std::cout << "First few output values: ";
    for (float val : h_out) {
        std::cout << val << " ";
    }
    std::cout << "\n";*/
}

// === Step 5: Benchmark Harness ===
void run_attention_benchmark() {
    allocate_and_init();

    auto time = Halide::Tools::benchmark(5, 5, [&]() {
        compute_scores();
        softmax_scores();
        compute_output();
        cudaDeviceSynchronize();
    });

    std::cout << "Runtime: " << time << " s\n";

    // TODO: (Optional) Copy output to host for correctness check
    // TODO: Free device memory and destroy descriptors
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