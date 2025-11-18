#include <torch/torch.h>
#include <iostream>

#include "halide_benchmark.h"

using namespace Halide::Tools;

// // -------------------------------------------------------------------
// // Built-in attention (nn::MultiheadAttention)
// // -------------------------------------------------------------------
// torch::Tensor attention_builtin(torch::Tensor Q, torch::Tensor K, torch::Tensor V) {
//     int64_t embed_dim = Q.size(-1);

//     // Create MHA with default layout (seq_len, batch, embed_dim)
//     auto mha = torch::nn::MultiheadAttention(
//         torch::nn::MultiheadAttentionOptions(embed_dim, 1)  // no batch_first
//     );

//     mha->to(Q.device(), Q.scalar_type());

//     // Convert [N, L, D] -> [L, N, D]
//     auto Q_t = Q.transpose(0, 1);
//     auto K_t = K.transpose(0, 1);
//     auto V_t = V.transpose(0, 1);

//     // Forward pass → output in fp16, accumulated internally in fp32
//     std::tuple<torch::Tensor, torch::Tensor> output;
//     auto time = benchmark(5, 5, [&]() {
//         output = mha->forward(Q_t, K_t, V_t);
//         torch::cuda::synchronize();
//     });
    
//     std::cout << "Runtime: " << std::fixed << std::setprecision(9) << time << "\n";

//     // Back to [N, L, D]
//     return std::get<0>(output);
// }

// -------------------------------------------------------------------
// Manual attention: scores = (QK^T) / sqrt(d); softmax; probs*V
// -------------------------------------------------------------------
torch::Tensor attention_manual(torch::Tensor Q, torch::Tensor K, torch::Tensor V) {
    auto d = Q.size(-1);
    auto scale = 1.0f / std::sqrt(static_cast<float>(d));

    // [N, L, D] × [N, D, L] -> [N, L, L]
    auto Q_f32 = Q.to(torch::kFloat);
    auto K_f32 = K.to(torch::kFloat);
    auto scores = torch::matmul(Q_f32, K_f32.transpose(-2, -1)) * scale;

    // softmax over keys
    auto probs = torch::softmax(scores, -1);

    // [N, L, L] × [N, L, D] -> [N, L, D]
    auto V_f32 = V.to(torch::kFloat);
    auto output = torch::matmul(probs, V_f32).to(torch::kFloat);

    return output;
}

int main() {
    torch::Device device(torch::kCUDA);
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

    // Random input (Q=K=V for simplicity)
    auto q = torch::randn({N, L, D},
        torch::TensorOptions().dtype(torch::kHalf).device(device));
    auto k = torch::randn({N, L, D},
        torch::TensorOptions().dtype(torch::kHalf).device(device));
    auto v = torch::randn({N, L, D},
        torch::TensorOptions().dtype(torch::kHalf).device(device));

    torch::Tensor out_manual;
    auto time = benchmark(5, 5, [&]() {
        out_manual = attention_manual(q, k, v);
        torch::cuda::synchronize();
    });

    std::cout << "Runtime: " << std::fixed << std::setprecision(9) << time << "\n";
      
    //std::cout << "Builtin output shape: " << out_builtin.sizes() << std::endl;
    std::cout << "Manual output shape:  " << out_manual.sizes() << std::endl;
    std::cout << "Output dtype: " << out_manual.dtype() << std::endl;

    return 0;
}
