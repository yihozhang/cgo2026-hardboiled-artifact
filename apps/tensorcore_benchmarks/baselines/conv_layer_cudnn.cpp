// baselines/conv_layer_cudnn.cpp
#include <cudnn.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>

#include "halide_benchmark.h"

using namespace Halide::Tools;

#define CHECK_CUDNN(expr)                                      \
    do {                                                       \
        cudnnStatus_t status = (expr);                         \
        if (status != CUDNN_STATUS_SUCCESS) {                  \
            std::cerr << "cuDNN error: "                       \
                      << cudnnGetErrorString(status) << "\n";  \
            std::exit(EXIT_FAILURE);                           \
        }                                                      \
    } while (0)

#define CHECK_CUDA(expr)                                       \
    do {                                                       \
        cudaError_t status = (expr);                           \
        if (status != cudaSuccess) {                           \
            std::cerr << "CUDA error: "                        \
                      << cudaGetErrorString(status) << "\n";   \
            std::exit(EXIT_FAILURE);                           \
        }                                                      \
    } while (0)

int main() {
    #ifndef NN_TENSOR_N
    const int N = 4096;
#else
    const int N = NN_TENSOR_N;
#endif

#ifndef NN_TENSOR_C
    const int C = 16;
#else
    const int C = NN_TENSOR_C;
#endif

#ifndef NN_TENSOR_H
    const int H = 64;
#else
    const int H = NN_TENSOR_H;
#endif

#ifndef NN_TENSOR_W
    const int W = 64;
#else
    const int W = NN_TENSOR_W;
#endif

#ifndef KERNEL_SIZE
    const int kSize = 3;
#else
    const int kSize = KERNEL_SIZE;
#endif
    const int pad_h = 1, pad_w = 1;
    const int stride_h = 1, stride_w = 1;

    cudnnHandle_t handle;
    CHECK_CUDNN(cudnnCreate(&handle));

    // Input descriptor (NHWC layout)
    cudnnTensorDescriptor_t inputDesc;
    CHECK_CUDNN(cudnnCreateTensorDescriptor(&inputDesc));
    CHECK_CUDNN(cudnnSetTensor4dDescriptor(
        inputDesc,
        CUDNN_TENSOR_NHWC,
        CUDNN_DATA_HALF,
        N, C, H, W));

    // Filter descriptor (NHWC layout too!)
    cudnnFilterDescriptor_t filterDesc;
    CHECK_CUDNN(cudnnCreateFilterDescriptor(&filterDesc));
    CHECK_CUDNN(cudnnSetFilter4dDescriptor(
        filterDesc, CUDNN_DATA_HALF, CUDNN_TENSOR_NHWC,
        C, C, kSize, kSize));

    // Convolution descriptor
    cudnnConvolutionDescriptor_t convDesc;
    CHECK_CUDNN(cudnnCreateConvolutionDescriptor(&convDesc));
    CHECK_CUDNN(cudnnSetConvolution2dDescriptor(
        convDesc,
        pad_h, pad_w,
        stride_h, stride_w,
        1, 1,
        CUDNN_CROSS_CORRELATION,
        CUDNN_DATA_FLOAT)); // accumulate in fp32 (fix)

    // Allow TensorCore math
    CHECK_CUDNN(cudnnSetConvolutionMathType(convDesc, CUDNN_TENSOR_OP_MATH_ALLOW_CONVERSION));

    // Output dims
    int outN, outC, outH, outW;
    CHECK_CUDNN(cudnnGetConvolution2dForwardOutputDim(
        convDesc, inputDesc, filterDesc, &outN, &outC, &outH, &outW));

    // Bias descriptor (NHWC, float)
    cudnnTensorDescriptor_t biasDesc;
    CHECK_CUDNN(cudnnCreateTensorDescriptor(&biasDesc));
    CHECK_CUDNN(cudnnSetTensor4dDescriptor(
        biasDesc,
        CUDNN_TENSOR_NHWC,
        CUDNN_DATA_HALF, // fix
        1, outC, 1, 1));

    // Output descriptor (NHWC, float)
    cudnnTensorDescriptor_t outputDesc;
    CHECK_CUDNN(cudnnCreateTensorDescriptor(&outputDesc));
    CHECK_CUDNN(cudnnSetTensor4dDescriptor(
        outputDesc,
        CUDNN_TENSOR_NHWC,
        CUDNN_DATA_HALF,
        outN, outC, outH, outW)); // (fix)

    // Allocate GPU memory
    size_t inputBytes  = N * C * H * W * sizeof(__half);
    size_t filterBytes = C * C * kSize * kSize * sizeof(__half);
    size_t biasBytes   = outC * sizeof(__half);
    size_t outputBytes = outN * outC * outH * outW * sizeof(__half);

    __half* d_input;  CHECK_CUDA(cudaMalloc(&d_input, inputBytes));
    __half* d_filter; CHECK_CUDA(cudaMalloc(&d_filter, filterBytes));
    __half*  d_bias;   CHECK_CUDA(cudaMalloc(&d_bias, biasBytes));
    __half*  d_output; CHECK_CUDA(cudaMalloc(&d_output, outputBytes));

    // Init host data
    std::vector<__half> h_input(N * C * H * W, __float2half(1.0f));
    std::vector<__half> h_filter(C * C * kSize * kSize, __float2half(1.0f));
    std::vector<__half> h_bias(C, __float2half(0.0f));

    CHECK_CUDA(cudaMemcpy(d_input, h_input.data(), inputBytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_filter, h_filter.data(), filterBytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_bias, h_bias.data(), biasBytes, cudaMemcpyHostToDevice));

    // Pick algorithm
    int maxAlgoCount = 8;
    std::vector<cudnnConvolutionFwdAlgoPerf_t> perf(maxAlgoCount);
    int returnedAlgoCount = 0;
    cudnnFindConvolutionForwardAlgorithm(
        handle, inputDesc, filterDesc, convDesc, outputDesc,
        maxAlgoCount, &returnedAlgoCount, perf.data());
    int bestIdx = -1;
    for (int i = 0; i < returnedAlgoCount; i++) {
        if (perf[i].mathType == CUDNN_TENSOR_OP_MATH_ALLOW_CONVERSION &&
            (bestIdx == -1 || perf[i].time < perf[bestIdx].time)) {
            bestIdx = i;
        }
    }

    cudnnConvolutionFwdAlgo_t algo = perf[bestIdx].algo;
    

    // Workspace
    size_t wsSize = 0;
    CHECK_CUDNN(cudnnGetConvolutionForwardWorkspaceSize(
        handle, inputDesc, filterDesc, convDesc, outputDesc, algo, &wsSize));
    void* d_workspace = nullptr;
    CHECK_CUDA(cudaMalloc(&d_workspace, wsSize));

    // Activation descriptor (ReLU)
    cudnnActivationDescriptor_t actDesc;
    CHECK_CUDNN(cudnnCreateActivationDescriptor(&actDesc));
    CHECK_CUDNN(cudnnSetActivationDescriptor(
        actDesc, CUDNN_ACTIVATION_RELU, CUDNN_PROPAGATE_NAN, 0.0));

    float alpha = 1.0f, beta = 0.0f;

    // Benchmark
     auto time = benchmark(100, 5, [&]() {
        CHECK_CUDNN(cudnnConvolutionBiasActivationForward(
            handle,
            &alpha,
            inputDesc, d_input,
            filterDesc, d_filter,
            convDesc, algo,
            d_workspace, wsSize,
            &beta,
            outputDesc, d_output,  // convolution output
            biasDesc, d_bias,      // bias
            actDesc,               // activation (ReLU)
            outputDesc, d_output   // final output
        ));
        cudaDeviceSynchronize();
    });

    std::cout << "Version: cuDNN" << "\n";
    std::cout << "Input: " << N << "x" << H << "x" << W << "x" << C << "\n";
    std::cout << "Runtime: "
              << std::fixed << std::setprecision(6)
              << time << std::endl;

    // Cleanup
    cudaFree(d_input);
    cudaFree(d_filter);
    cudaFree(d_output);
    cudaFree(d_bias);
    cudaFree(d_workspace);

    cudnnDestroyActivationDescriptor(actDesc);
    cudnnDestroyTensorDescriptor(inputDesc);
    cudnnDestroyTensorDescriptor(outputDesc);
    cudnnDestroyTensorDescriptor(biasDesc);
    cudnnDestroyFilterDescriptor(filterDesc);
    cudnnDestroyConvolutionDescriptor(convDesc);
    cudnnDestroy(handle);

    return 0;
}