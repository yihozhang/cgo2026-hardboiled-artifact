#include "HalideBuffer.h"
#include "HalideRuntime.h"
#include "HalideRuntimeCuda.h"
#include "halide_benchmark.h"

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

void resize_sim(float* input, float* output, int m, int n, int C, float scale_factor);

int main(int argc, char **argv) {

#ifdef _WIN32
    _putenv_s("HL_CUDA_JIT_MAX_REGISTERS", "256");
#else
    setenv("HL_CUDA_JIT_MAX_REGISTERS", "256", 1);
#endif

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
                        if (ky < kSize / 2 && kx < kSize / 2) {
                            expected += halide_float16_bits_to_float(kernel(2 * kx + (x & 1), 2 * ky + (y & 1))) * halide_float16_bits_to_float(image(x / 2 + kx, y / 2 + ky));
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
    Buffer<uint16_t> matA(K, M);
    for (int y = 0; y < M; y++) {
        for (int x = 0; x < K; x++) {
            matA(x, y) = float_to_float16(rand() & 1);  // uint16_t(rand() % 20);
        }
    }

    Buffer<uint16_t> matB(N, K);
    for (int y = 0; y < K; y++) {
        for (int x = 0; x < N; x++) {
            matB(x, y) = float_to_float16(rand() & 1);  // uint16_t(rand() % 20);
        }
    }

    matA.raw_buffer()->type = halide_type_t(halide_type_float, 16);
    matB.raw_buffer()->type = halide_type_t(halide_type_float, 16);

    // Create output buffer
    Buffer<float> output(M, N);

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
#elif defined(RUN_rec_filter)
    const int M = IMG_COL; // by default this is 1024*1024
    const int N = 2; // stereo audio

    std::string benchmark_name = BENCHMARK_NAME;

    std::cout << "Running " << benchmark_name << " with:" << std::endl;
    std::cout << "  Image size: " << N << "x" << M << std::endl;
    std::cout << "  Schedule: " << SCHEDULE << std::endl;

    // Create matrix buffers with random values
    Buffer<uint16_t> img(M, N);
    for (int y = 0; y < N; y++) {
        for (int x = 0; x < M; x++) {
            img(x, y) = float_to_float16(rand() % 2 == 0 ? 1.f : 0.0);  // uint16_t(rand() % 20);
        }
    }

    img.raw_buffer()->type = halide_type_t(halide_type_float, 16);

    int delay_factor = 16;
    int tile_width = 4096;
    bool is_tc = strcmp(SCHEDULE, "tensorcore") == 0;
    // Create output buffer
    Buffer<float> output = is_tc ? Buffer<float>(delay_factor, (tile_width/delay_factor), M/tile_width, N) : Buffer<float>(tile_width, M/tile_width, N);
    // TODO: I don't know why we need extent order + 2 instead of order + 1 for the coeff array
    Buffer<float> coeff(4);
    // float a = 0.9, b = -0.45;
    // float a = 1.6, b = -0.81;
    float a = -1, b = -1;
    coeff(0) = 0;
    coeff(1) = a;
    coeff(2) = b;
    coeff(3) = 0;
    // Call the generated function
    auto time = benchmark(5, 5, [&]() {
        // NB: Hardcode the coefficients for now
        rec_filter(img.raw_buffer(), coeff, output.raw_buffer());
        output.device_sync();
    });

    if (output.has_device_allocation()) {
        output.copy_to_host();
    }

    output.device_sync();

    std::cout << "Runtime: " << std::fixed << std::setprecision(9) << time << "\n";

    {
        int delay_factor = 16 * 1024;
        int order = 2;
        float aa[] = {0, a, b};
        float a1[3][delay_factor + 1] = {0.f};


        a1[0][0] = 1.;
        for (int o = 0; o < order; o++) {
            for (int i = 1; i <= delay_factor; i++) {
                for (int j = 1; j <= i; j++) {
                    if (j + o <= order) {
                        a1[o][i] += a1[0][i - j] * aa[j + o];
                    }
                }
                // std::cout << std::fixed << std::setprecision(10) << a1[o][i] << " ";
            }
            // std::cout << "\n";
        }
    }

    // Verify results
    if (VERIFY_OUTPUT) {
        float expected[100000][2] = {0.f};
        bool success = true;
        for (int y = 0; y < 2; y++) {
            if (!success) {
                break;
            }
            for (int x = 0; x < 100000; x++) {
                if (!success) {
                    break;
                }
                expected[x][y] = halide_float16_bits_to_float(img(x, y)) +
                    (x > 0 ? expected[x-1][y] * a : 0.f) + 
                    (x > 1 ? expected[x-2][y] * b : 0.f);
                auto o = is_tc ? output(x%delay_factor,(x%tile_width)/delay_factor, x/tile_width, y) : output(x%tile_width,x/tile_width, y);
                // std::cout << expected[x][y] << " " << o << " " << halide_float16_bits_to_float(img(x, y)) << "\n";
                if (fabs(expected[x][y] - o) > 0.01f) {
                    std::cerr << "Error at (" << x << ", " << y << "): "
                              << std::fixed << std::setprecision(10)
                              << o << " != " << expected[x][y] << "\n";
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
#elif defined(RUN_resize)
    // Create test data using compile-time definitions
    const int M = IMG_COL;
    const int N = IMG_ROW;
    const int C = 3;
    // const std::vector<float> scales = {1, 0.75, 1.5};
    const std::vector<float> scales = {0.1,0.25, 0.4, 0.5, 0.66, 0.75, 0.9, 0.99, 1.01, 1.1, 1.5, 2., 2.5, 4};
    std::string benchmark_name = BENCHMARK_NAME;

    std::cout << "Running " << benchmark_name << " with:" << std::endl;
    std::cout << "  Image size: " << N << "x" << M << std::endl;
    std::cout << "  Schedule: " << SCHEDULE << std::endl;

    for (const float scale: scales) {
        std::cout << "  scale: " << scale << std::endl;
        
        const int OM = M * scale;
        const int ON = N * scale;

        // Create matrix buffers with random values
        Buffer<float, 3> img(M, N, C);
        FOR (c, C) {
            FOR (y, N) {
                FOR (x, M) {
                    img(x, y, c) = float(rand() % 100) / 100.f;
                }
            }
        }

        // smallest multiple of 16 that is greater than OM and ON.
        const int OM_realized = (strcmp(SCHEDULE, "tensorcore") == 0) ? ((OM + 15) & ~15) : OM;
        const int ON_realized = (strcmp(SCHEDULE, "tensorcore") == 0) ? ((ON + 15) & ~15) : ON;
        Buffer<float, 3> output(OM_realized, ON_realized, C);
        auto resize = scale > 1.0f ? resize_up : resize_down;
        auto time = benchmark(5, 5, [&]() {
            resize(img.raw_buffer(), scale, output.raw_buffer());
            output.device_sync();
        });

        if (output.has_device_allocation()) {
            output.copy_to_host();
        }
        output.device_sync();

        std::cout << "Runtime: " << std::fixed << std::setprecision(9) << time << "\n";

        if (VERIFY_OUTPUT) {
            float* expected = new float[OM * ON * C];
            resize_sim((float *)img.raw_buffer()->host, expected, M, N, C, scale);
            
            // FOR (y, 16) {
            //     FOR (x, 16) {
            //         std::cout << std::fixed << std::setprecision(4) << img(x, y, 0) << " ";
            //     }
            //     std::cout << "\n";
            // }
            // std::cout << "\n";
            // FOR (y, 16) {
            //     FOR (x, 16) {
            //         std::cout << std::fixed << std::setprecision(4) << output(x, y, 0) << " ";
            //     }
            //     std::cout << "\n";
            // }
            // std::cout << "\n";
            // FOR (y, 16) {
            //     FOR (x, 16) {
            //         std::cout << std::fixed << std::setprecision(4) << expected[y * OM + x] << " ";
            //     }
            //     std::cout << "\n";
            // }
            bool success = true;
            FOR (c, C) {
                if (!success) break;
                FOR (y, ON) {
                    if (!success) break;
                    FOR (x, OM) {
                        float exp = expected[c * OM * ON + y * OM + x];
                        if (fabs(exp - output(x, y, c)) > 0.01f) {
                            std::cerr << "Error at (" << x << ", " << y << ", " << c << "): "
                                << std::fixed << std::setprecision(10)
                                << output(x, y, c) << " != " << exp << "\n";
                            success = false;
                            break;
                        }
                    }
                }
            }

            if (success) {
                // std::cout << "Outputs match!\n";
            } else {
                std::cout << "Outputs do not match...\n";
                return 1;
            }
        }
    }

#else
#error "Unknown benchmark type"
#endif

    return 0;
}

float sinc(float x) {
    x *= 3.14159265359f;
    return sin(x) / x;
}

float lanczos(float x) {
    float value = sinc(x) * sinc(x / 3);
    value = x == 0.0f ? 1.0f : value;        // Take care of singularity at zero
    value = (x > 3 || x < -3) ? 0.0f : value;  // Clamp to zero out of bounds
    return value;
}

float clamp(float x, float l, float h) {
    return std::min(std::max(x, l), h);
}

void resize_sim(float* input, float* output, int m, int n, int C, float scale_factor) {
    const int taps = 6;
    bool upsample = scale_factor > 1.0f;

    float inverse_scale_factor = 1.0f / scale_factor;

    float kernel_scaling = upsample ? 1.0f : scale_factor;
    float inverse_kernel_scaling = upsample ? 1.0f : inverse_scale_factor;

    float kernel_radius = 0.5f * taps * inverse_kernel_scaling;

    int kernel_taps = int(ceil(taps * inverse_kernel_scaling));

    int resized_m = int(m * scale_factor);
    int resized_n = int(n * scale_factor);

    float* kernel_x = new float[kernel_taps * resized_m];
    float* kernel_y = new float[kernel_taps * resized_n];

    FOR (x, resized_m) {
        float sum = 0.f;
        FOR (k, kernel_taps) {
            float sourcex = (x + 0.5f) * inverse_scale_factor - 0.5f;
            int beginx = int(ceil(sourcex - kernel_radius));
            // Compared to the original app, don't need to +1 here because max = min + extent - 1
            beginx = clamp(beginx, 0, m - kernel_taps);
            kernel_x[x * kernel_taps + k] = lanczos((k + beginx - sourcex) * kernel_scaling);
            sum += kernel_x[x * kernel_taps + k];
        }
        FOR (k, kernel_taps) {
            kernel_x[x * kernel_taps + k] /= sum;
        }
    }

    FOR (y, resized_n) {
        float sum = 0.f;
        FOR (k, kernel_taps) {
            float sourcey = (y + 0.5f) * inverse_scale_factor - 0.5f;
            int beginy = int(ceil(sourcey - kernel_radius));
            beginy = clamp(beginy, 0, n - kernel_taps);
            kernel_y[y * kernel_taps + k] = lanczos((k + beginy - sourcey) * kernel_scaling);
            sum += kernel_y[y * kernel_taps + k];
        }
        FOR (k, kernel_taps) {
            kernel_y[y * kernel_taps + k] /= sum;
        }
    }

    float* resized_y = new float[m * resized_n * C];

    FOR (c, C) {
        FOR (y, resized_n) {
            FOR (x, m) {
                resized_y[c * resized_n * m + y * m + x] = 0.f;
                FOR (r, kernel_taps) {
                    float sourcey = (y + 0.5f) * inverse_scale_factor - 0.5f;
                    int beginy = int(ceil(sourcey - kernel_radius));
                    beginy = clamp(beginy, 0, n - kernel_taps);
                    resized_y[c * resized_n * m + y * m + x] += 
                        kernel_y[y * kernel_taps + r] * 
                        input[c * n * m + (beginy + r) * m + x];
                }
            }
        }
    }

    FOR (c, C) {
        FOR (y, resized_n) {
            FOR (x, resized_m) {
                output[c * resized_n * resized_m + y * resized_m + x] = 0.f;
                FOR (r, kernel_taps) {
                    float sourcex = (x + 0.5f) * inverse_scale_factor - 0.5f;
                    int beginx = int(ceil(sourcex - kernel_radius));
                    beginx = clamp(beginx, 0, m - kernel_taps);
                    output[c * resized_n * resized_m + y * resized_m + x] += 
                        kernel_x[x * kernel_taps + r] * 
                        resized_y[c * resized_n * m + y * m + r + beginx];
                }
                output[c * resized_n * resized_m + y * resized_m + x] =
                    clamp(output[c * resized_n * resized_m + y * resized_m + x], 0.f, 1.f);
            }
        }
    }

    delete[] kernel_x;
    delete[] kernel_y;
    delete[] resized_y;
}