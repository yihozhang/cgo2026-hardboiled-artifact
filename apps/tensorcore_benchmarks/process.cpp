#include "HalideBuffer.h"
#include "HalideRuntime.h"
#include "HalideRuntimeCuda.h"
#include "halide_benchmark.h"

#include <cmath>
#include <cstdlib>  // for rand()
#include <iomanip>  // for std::fixed and std::setprecision
#include <iostream>

#ifndef BENCHMARK_HEADER
#error "BENCHMARK_HEADER must be defined"
#endif

#include BENCHMARK_HEADER

using namespace Halide::Runtime;
using namespace Halide::Tools;

float bfloat16_to_float(uint16_t b) {
    // Assume little-endian floats
    uint16_t bits[2] = {0, b};
    float ret;
    memcpy(&ret, bits, sizeof(float));
    return ret;
}

// Stole this from Halide's float16.h since I didn't find it in runtime
uint16_t float_to_float16(float value) {
    // Start by copying over the sign bit
    uint16_t bits = std::signbit(value) << 15;

    // Check for special values
    if (value == 0) {
        return bits;
    } else if (std::isnan(value)) {
        return bits | 0x7c00 | 0x03ff;
    } else if (std::isinf(value)) {
        return bits | 0x7c00;
    }

    int exp;
    // Get exponent, with bias already subtracted.
    std::frexp(value, &exp);
    if (exp > 16) {
        // Too large, return infinity. Per initialization, bits only
        // contains the sign bit, so this is +/-inf.
        return bits | 0x7c00;
    } else if (exp < -13) {
        // Too small, clamp to 2^-24
        value = std::ldexp(value, 24);
    } else {
        // Move the exponent from the float into the half.
        value = std::ldexp(value, 11 - exp);
        bits |= ((exp + 13) << 10);
    }

    // We've normalized value as much as possible. Put the integer
    // portion of it into the mantissa.
    float ival;
    float frac = std::modf(value, &ival);
    bits += (uint16_t)(std::abs((int)ival));

    // Now consider the fractional part. We round to nearest with ties
    // going to even.
    frac = std::abs(frac);
    bits += (frac > 0.5f) | ((frac == 0.5f) & bits);

    return bits;
}

int main(int argc, char **argv) {

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
    Buffer<uint16_t> image(imgW, imgH);
    for (int y = 0; y < imgH; y++) {
        for (int x = 0; x < imgW; x++) {
            image(x, y) = float_to_float16(rand() & 1);  // uint16_t(rand() % 20);
        }
    }

    // Create kernel buffer
    Buffer<uint16_t> kernel(kSize);
    for (int i = 0; i < kSize; i++) {
        kernel(i) = float_to_float16(i & 1);
    }

    image.raw_buffer()->type = halide_type_t(halide_type_float, 16);
    kernel.raw_buffer()->type = halide_type_t(halide_type_float, 16);

    // Create output buffer
    Buffer<float> output(imgW - kSize, imgH);

    // Call the generated function
    auto time = benchmark(5, 5, [&]() {
        conv1d(kernel.raw_buffer(), image.raw_buffer(), output.raw_buffer());
        output.device_sync();
    });

    if (output.has_device_allocation()) {
        output.copy_to_host();
    }

    output.device_sync();

    std::cout << "Runtime: " << std::fixed << std::setprecision(9) << time << "\n";

    // Verify results
    if (VERIFY_OUTPUT) {
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
                    expected += halide_float16_bits_to_float(kernel(k)) * halide_float16_bits_to_float(image(x + k, y));
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
    Buffer<uint16_t> image(imgW, imgH);
    for (int y = 0; y < imgH; y++) {
        for (int x = 0; x < imgW; x++) {
            image(x, y) = float_to_float16(rand() & 1);  // uint16_t(rand() % 20);
        }
    }

    // Create kernel buffer
    Buffer<uint16_t> kernel(kSize, kSize);
    for (int i = 0; i < kSize; i++) {
        for (int j = 0; j < kSize; j++) {
            kernel(i, j) = float_to_float16((i ^ j) & 1);
        }
    }

    image.raw_buffer()->type = halide_type_t(halide_type_float, 16);
    kernel.raw_buffer()->type = halide_type_t(halide_type_float, 16);

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
        fn(kernel.raw_buffer(), image.raw_buffer(), output.raw_buffer());
        output.device_sync();
    });

    if (output.has_device_allocation()) {
        output.copy_to_host();
    }

    output.device_sync();

    std::cout << "Runtime: " << std::fixed << std::setprecision(9) << time << "\n";

    // Verify results
    if (VERIFY_OUTPUT) {
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
                        expected += halide_float16_bits_to_float(kernel(kx, ky)) * halide_float16_bits_to_float(image(x + kx, y + ky));
#elif defined(RUN_downsample)
                        expected += halide_float16_bits_to_float(kernel(kx, ky)) * halide_float16_bits_to_float(image(2 * x + kx, 2 * y + ky));
#else
                        expected += halide_float16_bits_to_float(kernel(kx, ky)) * halide_float16_bits_to_float(image(x / 2 + kx, y / 2 + ky));
#endif
                    }
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
    Buffer<uint16_t> matA(M, K);
    for (int y = 0; y < M; y++) {
        for (int x = 0; x < K; x++) {
            matA(x, y) = float_to_float16(rand() & 1);  // uint16_t(rand() % 20);
        }
    }

    Buffer<uint16_t> matB(K, N);
    for (int y = 0; y < K; y++) {
        for (int x = 0; x < N; x++) {
            matB(x, y) = float_to_float16(rand() & 1);  // uint16_t(rand() % 20);
        }
    }

    matA.raw_buffer()->type = halide_type_t(halide_type_float, 16);
    matB.raw_buffer()->type = halide_type_t(halide_type_float, 16);

    // Create output buffer
    Buffer<float> output(M, N);

#ifdef _WIN32
    _putenv_s("HL_CUDA_JIT_MAX_REGISTERS", "256");
#else
    setenv("HL_CUDA_JIT_MAX_REGISTERS", "256", 1);
#endif
    
    // Call the generated function
    auto time = benchmark(5, 5, [&]() {
        matmul(matA.raw_buffer(), matB.raw_buffer(), output.raw_buffer());
        output.device_sync();
    });

    if (output.has_device_allocation()) {
        output.copy_to_host();
    }

    output.device_sync();

    std::cout << "Runtime: " << std::fixed << std::setprecision(9) << time << "\n";

    // Verify results
    if (VERIFY_OUTPUT) {
        bool success = true;
        for (int y = 0; y < 256; y++) {
            if (!success) {
                break;
            }
            for (int x = 0; x < 256; x++) {
                if (!success) {
                    break;
                }
                float expected = 0.0f;
                for (int k = 0; k < K; k++) {
                    expected += halide_float16_bits_to_float(matA(k, y)) * halide_float16_bits_to_float(matB(x, k));
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
#else
#error "Unknown benchmark type"
#endif

    return 0;
}
