#include <arrayfire.h>
#include <iostream>
#include <chrono>
#include <vector>
#include <limits>
#include <numeric>

#include "conv_benchmarks.h"

int main() {
    std::vector<std::pair<int, int>> resolutions = {
        {3840, 2160},     // 4K
        {7680, 4320},     // 8K
        {15360, 8640},    // 16K
        {30720, 17280},   // 32K
        //{61440, 34560}    // 64K
    };

    std::vector<int> conv1d_kernels = { 64, 128 };
    std::vector<std::pair<int, int>> conv2d_kernels = {
        {16, 16},
        {32, 32}
    };

    for (auto [W, H] : resolutions) {
        for (int kw : conv1d_kernels) {
            std::cout << "\n[AF 1D] " << W << "x" << H << ", kernel: " << kw << "\n";
            
            try {
                af_conv1d(W, H, kw);
            }
            catch (...) {
                std::cout << "Failed.\n";
            }
            

            std::cout << "\n[OCV 1D] " << W << "x" << H << ", kernel: " << kw << "\n";
            try {
                ocv_conv1d(W, H, kw);
            }
            catch (...) {
                std::cout << "Failed.\n";
            }
        }

        for (auto [kw, kh] : conv2d_kernels) {
            std::cout << "\n[AF 2D] " << W << "x" << H << ", kernel: " << kw << "x" << kh << "\n";
            try {
                if (kw == 32) throw std::runtime_error("");
                af_conv2d(W, H, kw, kh);
            }
            catch (...) {
                std::cout << "Failed.\n";
            }

            std::cout << "\n[OCV 2D] " << W << "x" << H << ", kernel: " << kw << "x" << kh << "\n";
            try {
                ocv_conv2d(W, H, kw, kh);
            }
            catch (...) {
                std::cout << "Failed.\n";
            }
        }
    }

    return 0;
}