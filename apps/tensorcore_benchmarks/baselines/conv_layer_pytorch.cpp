#include <torch/torch.h>
#include <iostream>

#include "halide_benchmark.h"

using namespace Halide::Tools;

int main() {
    torch::Device device(torch::kCUDA);
    
    const int N = 4096, C_in = 32, C_out = 32, H = 64, W = 64;

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