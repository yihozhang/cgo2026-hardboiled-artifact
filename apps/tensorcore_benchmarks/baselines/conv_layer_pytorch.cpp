#include <torch/torch.h>
#include <iostream>

#include "halide_benchmark.h"

using namespace Halide::Tools;

int main() {
    torch::Device device(torch::kCUDA);
#ifndef NN_TENSOR_N
    const int N = 4096;
#else
    const int N = NN_TENSOR_N;
#endif

#ifndef NN_TENSOR_C
    const int C_in = 16, C_out = 16;
#else
    const int C_in = NN_TENSOR_C, C_out = NN_TENSOR_C;
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

    // Input: half precision, channels_last
    auto input = torch::randn({N, C_in, H, W},
        torch::TensorOptions().dtype(torch::kHalf).device(device))
        .contiguous(torch::MemoryFormat::ChannelsLast);

    // Conv: weights half, bias float
    auto conv = torch::nn::Conv2d(
        torch::nn::Conv2dOptions(C_in, C_out, 3)
            .padding(1)
            .stride(1)
            .bias(true));

    // after constructing conv
    conv->to(device, torch::kHalf);
    
    // Initialize weights/bias
    {
        torch::NoGradGuard no_grad;

        // Weights (fp16)
        auto filter = torch::ones({C_out, C_in, 3, 3},
            torch::TensorOptions().dtype(torch::kHalf).device(device));
        conv->weight.copy_(filter);

        // Bias (fp16 — matches module dtype)
        auto bias = torch::randn({C_out},
            torch::TensorOptions().dtype(torch::kHalf).device(device));
        conv->bias.copy_(bias);
    }

    // Forward pass → output in fp16, accumulated internally in fp32
    torch::Tensor output;
    auto time = benchmark(5, 5, [&]() {
        output = torch::relu(conv->forward(input));
        torch::cuda::synchronize();
    });

    std::cout << "Version: Pytorch" << "\n";
    std::cout << "Input: " << N << "x" << H << "x" << W << "x" << C_in << "\n";
    std::cout << "Runtime: " << std::fixed << std::setprecision(9) << time << "\n";
}
