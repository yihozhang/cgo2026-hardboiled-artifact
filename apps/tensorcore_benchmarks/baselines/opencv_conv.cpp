#include <opencv2/opencv.hpp>
#include <opencv2/cudafilters.hpp>
#include <chrono>
#include <iostream>
#include <numeric>

#include "conv_benchmarks.h"

void ocv_conv1d(int W, int H, int KW) {
    const int sets = 5;
    const int runs_per_set = 5;

    // std::cout << "Allocating input and kernel...\n";
    cv::Mat input(H, W, CV_32F);
    cv::randu(input, 0.0f, 1.0f);

    cv::Mat kernel = cv::Mat::ones(1, KW, CV_32F);

    // std::cout << "Uploading to GPU...\n";
    cv::cuda::GpuMat d_input(input);


    cv::Mat output(H, W, CV_32F);
    cv::cuda::GpuMat d_output(input);
    cv::Ptr<cv::cuda::Filter> filter = cv::cuda::createLinearFilter(d_input.type(), d_output.type(), kernel, cv::Point(-1, -1), 0);

    // Warmup
    // std::cout << "Running warmup...\n";
    filter->apply(d_input, d_output);

    std::vector<double> fastest_times;

    // std::cout << "Benchmarking OpenCV CUDA Conv1D...\n";
    for (int s = 0; s < sets; ++s) {
        double min_time = std::numeric_limits<double>::max();

        for (int r = 0; r < runs_per_set; ++r) {
            auto start = std::chrono::high_resolution_clock::now();
            filter->apply(d_input, d_output);
            auto end = std::chrono::high_resolution_clock::now();
            double time_ms = std::chrono::duration<double, std::milli>(end - start).count();
            min_time = std::min(min_time, time_ms);
        }
        d_output.download(output);

        fastest_times.push_back(min_time);
    }

    double avg_time = std::accumulate(fastest_times.begin(), fastest_times.end(), 0.0) / fastest_times.size();
    std::cout << "Average of 5 fastest runs: " << avg_time / 1000.0 << " s\n";
    
    input.release();
    kernel.release();
    output.release();
}

void ocv_conv2d(int W, int H, int KW, int KH) {
    const int sets = 5;
    const int runs_per_set = 5;

    // std::cout << "Allocating input and kernel...\n";
    cv::Mat input(H, W, CV_32F);
    cv::randu(input, 0.0f, 1.0f);

    cv::Mat kernel = cv::Mat::ones(KH, KW, CV_32F);

    // std::cout << "Uploading to GPU...\n";
    cv::cuda::GpuMat d_input(input);

    
    cv::Mat output(H, W, CV_32F);
    cv::cuda::GpuMat d_output(input);
    cv::Ptr<cv::cuda::Filter> filter = cv::cuda::createLinearFilter(d_input.type(), d_output.type(), kernel, cv::Point(-1, -1), 0);

    // Warmup
    // std::cout << "Running warmup...\n";
    filter->apply(d_input, d_output);

    std::vector<double> fastest_times;

    // std::cout << "Benchmarking OpenCV CUDA Conv2D...\n";
    for (int s = 0; s < sets; ++s) {
        double min_time = std::numeric_limits<double>::max();

        for (int r = 0; r < runs_per_set; ++r) {
            auto start = std::chrono::high_resolution_clock::now();
            filter->apply(d_input, d_output);
            auto end = std::chrono::high_resolution_clock::now();
            double time_ms = std::chrono::duration<double, std::milli>(end - start).count();
            min_time = std::min(min_time, time_ms);
        }
        d_output.download(output);

        fastest_times.push_back(min_time);
    }

    double avg_time = std::accumulate(fastest_times.begin(), fastest_times.end(), 0.0) / fastest_times.size();
    std::cout << "Average of 5 fastest runs: " << avg_time / 1000.0 << " s\n";

    input.release();
    kernel.release();
    output.release();
}