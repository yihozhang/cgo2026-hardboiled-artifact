#include <arrayfire.h>
#include <iostream>
#include <chrono>
#include <vector>
#include <limits>
#include <numeric>

#include "conv_benchmarks.h"

#ifdef max
#undef max
#endif

#ifdef min
#undef min
#endif

void af_conv1d(int W, int H, int KW) {
    af::setBackend(AF_BACKEND_CUDA); // force GPU backend

    // std::cout << "Allocating image (" << H << "x" << W << ")...\n";
    af::array input = af::randu(H * W, f32);

    // std::cout << "Creating kernel (" << KW << ")...\n";
    af::array kernel = af::randu(KW, f32);

    // std::cout << "Running warmup convolution...\n";
    af::array output = af::convolve1(input, kernel, AF_CONV_EXPAND, AF_CONV_SPATIAL);

    input.eval();   // Forces allocation and transfer to GPU
    kernel.eval();  // Same for the kernel
    output.eval();
    input.device<void>();
    kernel.device<void>();
    output.device<void>();

    af::sync(); // Ensure warmup finishes

    // Benchmark
    const int sets = 5;
    const int runs_per_set = 5;

    std::vector<double> fastest_times;

    // std::cout << "Benchmarking Conv1D: " << sets << " sets (each with " << runs_per_set << " runs)...\n";

    for (int s = 0; s < sets; ++s) {
        double min_time = std::numeric_limits<double>::max();

        for (int r = 0; r < runs_per_set; ++r) {
            auto start = std::chrono::high_resolution_clock::now();
            af::array output = af::convolve1(input, kernel, AF_CONV_EXPAND, AF_CONV_SPATIAL);
            af::sync(); // wait for GPU work to finish
            auto end = std::chrono::high_resolution_clock::now();

            double time_ms = std::chrono::duration<double, std::milli>(end - start).count();
            min_time = std::min(min_time, time_ms);
        }

        fastest_times.push_back(min_time);
    }

    // Final average of the 5 fastest runs (1 per set)
    double avg_time = std::accumulate(fastest_times.begin(), fastest_times.end(), 0.0) / fastest_times.size();
    std::cout << "Average of 5 fastest runs: " << avg_time / 1000.0 << " s\n";

    af::sync();         // Final sync
    af::deviceGC();     // Force garbage collection (deallocate unused device memory)
}

void af_conv2d(int W, int H, int KW, int KH) {
    af::setBackend(AF_BACKEND_CUDA); // force GPU backend

    // std::cout << "Allocating image (" << H << "x" << W << ")...\n";
    af::array input = af::randu(H, W, f32);

    // std::cout << "Creating kernel (" << KH << "x" << KW << ")...\n";
    af::array kernel = af::randu(KH, KW, f32);

    // std::cout << "Running warmup convolution...\n";
    af::array output = af::convolve2(input, kernel, AF_CONV_EXPAND, AF_CONV_SPATIAL);

    input.eval();   // Forces allocation and transfer to GPU
    kernel.eval();  // Same for the kernel
    output.eval();
    input.device<void>();
    kernel.device<void>();
    output.device<void>();

    af::sync(); // Ensure warmup finishes

    // Benchmark
    const int sets = 5;
    const int runs_per_set = 5;

    std::vector<double> fastest_times;

    //std::cout << "Benchmarking Conv2D: " << sets << " sets (each with " << runs_per_set << " runs)...\n";

    for (int s = 0; s < sets; ++s) {
        double min_time = std::numeric_limits<double>::max();

        for (int r = 0; r < runs_per_set; ++r) {
            auto start = std::chrono::high_resolution_clock::now();
            af::array output = af::convolve2(input, kernel, AF_CONV_EXPAND, AF_CONV_SPATIAL);
            af::sync(); // wait for GPU work to finish
            auto end = std::chrono::high_resolution_clock::now();

            double time_ms = std::chrono::duration<double, std::milli>(end - start).count();
            min_time = std::min(min_time, time_ms);
        }

        fastest_times.push_back(min_time);
    }

    // Final average of the 5 fastest runs (1 per set)
    double avg_time = std::accumulate(fastest_times.begin(), fastest_times.end(), 0.0) / fastest_times.size();
    std::cout << "Average of 5 fastest runs: " << avg_time / 1000.0 << " s\n";

    af::sync();         // Final sync
    af::deviceGC();     // Force garbage collection (deallocate unused device memory)
}