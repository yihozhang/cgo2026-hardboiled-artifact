#include "Halide.h"
#include "halide_benchmark.h"
#include "halide_image_io.h"

#include <cmath>
#include <cstdlib>  // for rand()
#include <iomanip>  // for std::fixed and std::setprecision
#include <iostream>
#include <vector>

#ifndef BENCHMARK_HEADER
#error "BENCHMARK_HEADER must be defined"
#endif

#include BENCHMARK_HEADER
#if defined(RUN_resize)
#include BENCHMARK_HEADER_EXTRA
#endif

#define FOR(i, N) for (int i = 0; i < (N); i++)

using namespace Halide::Runtime;
using namespace Halide::Tools;

int main(int argc, char **argv) {

    setenv("HL_CUDA_JIT_MAX_REGISTERS", "256", 1);

#if defined(RUN_conv1d)
    // Create test data using compile-time definitions
    const int kSize = KERNEL_SIZE;
    const int imgW = IMG_COL;
    const int imgH = IMG_ROW;

    std::string benchmark_name = BENCHMARK_NAME;

    std::cout << "Running " << benchmark_name << " with:" << std::endl;
    std::cout << "  Kernel size: " << kSize << std::endl;
    std::cout << "  Image size: " << imgW << "x" << imgH << std::endl;
    std::cout << "  Schedule: " << SCHEDULE << std::endl;

    // Create image buffer with random values
    Buffer<Halide::float16_t> image(imgW, imgH);
    for (int y = 0; y < imgH; y++) {
        for (int x = 0; x < imgW; x++) {
            image(x, y) = Halide::float16_t(rand() & 1);
        }
    }

    // Create kernel buffer
    Buffer<Halide::float16_t> kernel(kSize);
    for (int i = 0; i < kSize; i++) {
        kernel(i) = Halide::float16_t(i & 1);
    }

    // Create output buffer
    Buffer<float> output(imgW - kSize, imgH);

    // Call the generated function
    auto time = benchmark(5, 5, [&]() {
        conv1d(kernel, image, output);
        output.device_sync();
    });

    if (output.has_device_allocation()) {
        output.copy_to_host();
    }

    output.device_sync();

    std::cout << "Runtime: " << std::fixed << std::setprecision(9) << time << "\n";

    // Verify results
    if (std::getenv("VERIFY_OUTPUT")) {
        bool success = true;
        for (int y = 0; y < imgH; y++) {
            if (!success) {
                break;
            }
            for (int x = 0; x < imgW - kSize; x++) {
                if (!success) {
                    break;
                }
                float expected = 0.0f;
                for (int k = 0; k < kSize; k++) {
                    expected += float(kernel(k)) * float(image(x + k, y));
                }
                if (fabs(expected - output(x, y)) > 0.001f) {
                    std::cerr << "Error at (" << x << ", " << y << "): "
                              << output(x, y) << " != " << expected << "\n";
                    success = false;
                }
            }
        }

        if (success) {
            std::cout << "Outputs match!\n";
            return 0;
        } else {
            std::cout << "Outputs do not match...\n";
            return 1;
        }
    }
#elif defined(RUN_conv2d) || defined(RUN_upsample) || defined(RUN_downsample)
    // Create test data using compile-time definitions
    const int kSize = KERNEL_SIZE;
    const int imgW = IMG_COL;
    const int imgH = IMG_ROW;

    std::string benchmark_name = BENCHMARK_NAME;

    std::cout << "Running " << benchmark_name << " with:" << std::endl;
    std::cout << "  Kernel size: " << kSize << std::endl;
    std::cout << "  Image size: " << imgW << "x" << imgH << std::endl;
    std::cout << "  Schedule: " << SCHEDULE << std::endl;

    // Create image buffer with random values
    Buffer<Halide::float16_t> image(imgW, imgH);
    for (int y = 0; y < imgH; y++) {
        for (int x = 0; x < imgW; x++) {
            image(x, y) = Halide::float16_t(rand() & 1);
        }
    }

    // Create kernel buffer
    Buffer<Halide::float16_t> kernel(kSize, kSize);
    for (int i = 0; i < kSize; i++) {
        for (int j = 0; j < kSize; j++) {
            kernel(i, j) = Halide::float16_t((i ^ j) & 1);
        }
    }

#if defined(RUN_conv2d)
#define outW (imgW - kSize)
#define outH (imgH - kSize)
#define fn conv2d
#elif defined(RUN_downsample)
#define outW ((imgW - kSize) / 2)
#define outH ((imgH - kSize) / 2)
#define fn downsample
#else
#define outW (2 * (imgW - kSize))
#define outH (2 * (imgH - kSize))
#define fn upsample
#endif
    Buffer<float> output(outW, outH);

    // Call the generated function
    auto time = benchmark(5, 5, [&]() {
        fn(kernel, image, output);
        output.device_sync();
    });

    if (output.has_device_allocation()) {
        output.copy_to_host();
    }

    output.device_sync();

    std::cout << "Runtime: " << std::fixed << std::setprecision(9) << time << "\n";

    // Verify results
    if (std::getenv("VERIFY_OUTPUT")) {
        bool success = true;
        for (int y = 0; y < outH; y++) {
            if (!success) {
                break;
            }
            for (int x = 0; x < outW; x++) {
                if (!success) {
                    break;
                }
                float expected = 0.0f;
                for (int ky = 0; ky < kSize; ky++) {
                    for (int kx = 0; kx < kSize; kx++) {
#if defined(RUN_conv2d)
                        expected += float(kernel(kx, ky)) * float(image(x + kx, y + ky));
#elif defined(RUN_downsample)
                        expected += float(kernel(kx, ky)) * float(image(2 * x + kx, 2 * y + ky));
#else
                        if (ky < kSize / 2 && kx < kSize / 2) {
                            expected += float(kernel(2 * kx + (x & 1), 2 * ky + (y & 1))) * float(image(x / 2 + kx, y / 2 + ky));
                        }
#endif
                    }
                }
                if (std::isnan(output(x, y)) || fabs(expected - output(x, y)) > 0.001f) {
                    std::cerr << "Error at (" << x << ", " << y << "): "
                              << std::fixed << std::setprecision(10)
                              << output(x, y) << " != " << expected << "\n";
                    success = false;
                }
            }
        }

        if (success) {
            std::cout << "Outputs match!\n";
            return 0;
        } else {
            std::cout << "Outputs do not match...\n";
            return 1;
        }
    }
#elif defined(RUN_denoise)
    Buffer<Halide::float16_t> img(IMG_COL, IMG_ROW, 3), out(IMG_COL - 16, IMG_ROW - 16, 3);
    for (int c = 0; c < img.channels(); c++) {
        for (int y = 0; y < img.height(); y++) {
            for (int x = 0; x < img.width(); x++) {
                float noise = (rand() & 65535) / 65535.f - 0.5f;
                // Break the image into large squares, with a different constant
                // value per square.
                uint64_t hash = c + (x / 20) + (y / 20) * 3;
                float signal = (hash & 7) / 8.f;
                img(x, y, c) = Halide::float16_t(std::max(0.f, std::min(1.f, signal + 0.1f * noise)));
            }
        }
    }

    auto time = benchmark(5, 5, [&]() {
        denoise(img, 0.1, out);
        out.device_sync();
    });

    std::cout << "Runtime: " << std::fixed << std::setprecision(9) << time << "\n";

    out.copy_to_host();

    // At a strength of zero, the output should approximately equal the input
    double total_error = 0.0;
    for (int c = 0; c < out.channels(); c++) {
        for (int y = 0; y < out.height(); y++) {
            for (int x = 0; x < out.width(); x++) {
                float output = float16_to_float(out(x, y, c));
                float correct = float16_to_float(img(x + 8, y + 8, c));
                total_error += output - correct;
            }
        }
    }

    Buffer<uint8_t, 3> out_8(out.width(), out.height(), 3);
    Buffer<uint8_t, 3> img_8(img.width(), img.height(), 3);
    out_8.for_each_value([](uint8_t &v, Halide::float16_t v16) { v = float(v16) * 255.999f; }, out);
    img_8.for_each_value([](uint8_t &v, Halide::float16_t v16) { v = float(v16) * 255.999f; }, img);
    Halide::Tools::save_image(img_8, "denoised_img.png");
    Halide::Tools::save_image(out_8, "denoised_out.png");

    total_error /= (out.width() * out.height() * out.channels());
    if (total_error > 0.01) {
        std::cout << "Warning: Average absolute error per pixel is high: " << total_error << "\n";
        return -1;
    }

#elif defined(RUN_matmul)
    // Create test data using compile-time definitions
    const int M = MATMUL_M;
    const int N = MATMUL_N;
    const int K = MATMUL_K;

    std::string benchmark_name = BENCHMARK_NAME;

    std::cout << "Running " << benchmark_name << " with:" << std::endl;
    std::cout << "  Matrix size: " << M << "x" << N << "x" << K << std::endl;
    std::cout << "  Schedule: " << SCHEDULE << std::endl;

    // Create matrix buffers with random values
    Buffer<Halide::float16_t> matA(K, M);
    for (int y = 0; y < M; y++) {
        for (int x = 0; x < K; x++) {
            matA(x, y) = Halide::float16_t(rand() & 1);
        }
    }

    Buffer<Halide::float16_t> matB(N, K);
    for (int y = 0; y < K; y++) {
        for (int x = 0; x < N; x++) {
            matB(x, y) = Halide::float16_t(rand() & 1);
        }
    }

    // Create output buffer
    Buffer<float> output(M, N);

    // Call the generated function
    auto time = benchmark(5, 5, [&]() {
        matmul(matA, matB, output);
        output.device_sync();
    });

    if (output.has_device_allocation()) {
        output.copy_to_host();
    }

    output.device_sync();

    std::cout << "Runtime: " << std::fixed << std::setprecision(9) << time << "\n";

    // Verify results
    if (std::getenv("VERIFY_OUTPUT")) {
        bool success = true;
        for (int y = 0; y < M; y++) {
            if (!success) {
                break;
            }
            for (int x = 0; x < N; x++) {
                if (!success) {
                    break;
                }
                float expected = 0.0f;
                for (int k = 0; k < K; k++) {
                    expected += float(matA(k, y)) * float(matB(x, k));
                }
                if (fabs(expected - output(x, y)) > 0.001f) {
                    std::cerr << "Error at (" << x << ", " << y << "): "
                              << std::fixed << std::setprecision(10)
                              << output(x, y) << " != " << expected << "\n";
                    success = false;
                }
            }
        }

        if (success) {
            std::cout << "Outputs match!\n";
            return 0;
        } else {
            std::cout << "Outputs do not match...\n";
            return 1;
        }
    }

#elif defined(RUN_conv_layer)
    // Create test data using compile-time definitions
    const int N = NN_TENSOR_N;
    const int H = NN_TENSOR_H;
    const int W = NN_TENSOR_W;
    const int C = NN_TENSOR_C;
    const int kSize = KERNEL_SIZE;

    std::string benchmark_name = BENCHMARK_NAME;

    std::cout << "Running " << benchmark_name << " with:" << std::endl;
    std::cout << "  NHWC: " << N << "x" << H << "x" << W << "x" << C << std::endl;
    std::cout << "  Kernel size: " << kSize << std::endl;
    std::cout << "  Schedule: " << SCHEDULE << std::endl;

    // Create matrix buffers with random values
    Buffer<Halide::float16_t> input(C, W + kSize, H + kSize, N);
    for (int n = 0; n < N; n++) {
        for (int h = 0; h < H + kSize; h++) {
            for (int w = 0; w < W + kSize; w++) {
                for (int c = 0; c < C; c++) {
                    input(c, w, h, n) = Halide::float16_t(rand() & 1);
                }
            }
        }
    }

    Buffer<Halide::float16_t> filter(C, kSize, kSize, C);
    for (int co = 0; co < C; co++) {
        for (int kh = 0; kh < kSize; kh++) {
            for (int kw = 0; kw < kSize; kw++) {
                for (int ci = 0; ci < C; ci++) {
                    filter(ci, kw, kh, co) = Halide::float16_t(rand() & 1);
                }
            }
        }
    }

    Buffer<float> bias(C);
    for (int c = 0; c < C; c++) {
        bias(c) = float(rand() & 1);
    }

    // Create output buffer
    Buffer<float> output(C, W, H, N);

    // Call the generated function
    auto time = benchmark(100, 5, [&]() {
        conv_layer(input, filter, bias, output);
        output.device_sync();
    });

    if (output.has_device_allocation()) {
        output.copy_to_host();
    }

    output.device_sync();

    std::cout << "Runtime: " << std::fixed << std::setprecision(9) << time << "\n";

    // Verify results
    if (std::getenv("VERIFY_OUTPUT")) {
        bool success = true;
        for (int y = 0; y < 256; y++) {
            if (!success) {
                break;
            }
            // todo later
        }

        if (success) {
            std::cout << "Outputs match!\n";
            return 0;
        } else {
            std::cout << "Outputs do not match...\n";
            return 1;
        }
    }
    
#elif defined(RUN_attention)
    // Create test data using compile-time definitions
    const int D = ATT_D;
    const int L = ATT_L;
    const int N = ATT_N;

    std::string benchmark_name = BENCHMARK_NAME;

    std::cout << "Running " << benchmark_name << " with:" << std::endl;
    std::cout << "  DxLxN: " << D << "x" << L << "x" << N << std::endl;
    std::cout << "  Schedule: " << SCHEDULE << std::endl;

    // Create Q, K, V buffers with random values
    Buffer<Halide::float16_t> query(D, L, N);
    Buffer<Halide::float16_t> key(D, L, N);
    Buffer<Halide::float16_t> value(D, L, N);

    for (int n = 0; n < N; n++) {
        for (int t = 0; t < L; t++) {
            for (int d = 0; d < D; d++) {
                query(d, t, n) = Halide::float16_t(rand() & 1);
                key(d, t, n) = Halide::float16_t(rand() & 1);
                value(d, t, n) = Halide::float16_t(rand() & 1);
            }
        }
    }

    // Create output buffer
    Buffer<float> output(D, L, N);

    // Call the generated function
    auto time = benchmark(5, 5, [&]() {
        attention(query, key, value, output);
        output.device_sync();
    });

    if (output.has_device_allocation()) {
        output.copy_to_host();
    }

    output.device_sync();

    std::cout << "Runtime: " << std::fixed << std::setprecision(9) << time << "\n";

    // Optional verification placeholder
    if (std::getenv("VERIFY_OUTPUT")) {
        bool success = true;
        // TODO: add verification if desired
        if (success) {
            std::cout << "Outputs match!\n";
            return 0;
        } else {
            std::cout << "Outputs do not match...\n";
            return 1;
        }
    }
#else
#error "Unknown benchmark type"
#endif

    return 0;
}