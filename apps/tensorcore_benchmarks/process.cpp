#include "HalideBuffer.h"
#include "HalideRuntime.h"
#include "HalideRuntimeCuda.h"
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

float bfloat16_to_float(uint16_t b) {
    // Assume little-endian floats
    uint16_t bits[2] = {0, b};
    float ret;
    memcpy(&ret, bits, sizeof(float));
    return ret;
}

float float16_to_float(uint16_t f) {
    return halide_float16_bits_to_float(f);
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

void resize_sim(uint16_t *input, uint16_t *output, int m, int n, int C, float scale_factor);

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
                    expected += float16_to_float(kernel(k)) * float16_to_float(image(x + k, y));
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
                        expected += float16_to_float(kernel(kx, ky)) * float16_to_float(image(x + kx, y + ky));
#elif defined(RUN_downsample)
                        expected += float16_to_float(kernel(kx, ky)) * float16_to_float(image(2 * x + kx, 2 * y + ky));
#else
                        if (ky < kSize / 2 && kx < kSize / 2) {
                            expected += float16_to_float(kernel(2 * kx + (x & 1), 2 * ky + (y & 1))) * float16_to_float(image(x / 2 + kx, y / 2 + ky));
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
                    expected += float16_to_float(matA(k, y)) * float16_to_float(matB(x, k));
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
    const int M = IMG_COL;  // by default this is 1024*1024
    const int N = 2;        // stereo audio

    std::string benchmark_name = BENCHMARK_NAME;

    std::cout << "Running " << benchmark_name << " with:" << std::endl;
    std::cout << "  Image size: " << N << "x" << M << std::endl;
    std::cout << "  Schedule: " << SCHEDULE << std::endl;

    // Create matrix buffers with random values
    Buffer<uint16_t> img(M, N);
    for (int y = 0; y < N; y++) {
        for (int x = 0; x < M; x++) {
            img(x, y) = float_to_float16(rand() & 1);  // uint16_t(rand() % 20);
        }
    }

    img.raw_buffer()->type = halide_type_t(halide_type_float, 16);

    // Create output buffer
    Buffer<float> output(16, M / 16, N);

    float a = 0.9, b = -0.45;
    // Call the generated function
    auto time = benchmark(5, 5, [&]() {
        // NB: Hardcode the coefficients for now
        rec_filter(img.raw_buffer(), 0, a, b, output.raw_buffer());
        output.device_sync();
    });

    if (output.has_device_allocation()) {
        output.copy_to_host();
    }

    output.device_sync();

    std::cout << "Runtime: " << std::fixed << std::setprecision(9) << time << "\n";

    {
        int delay_factor = 16;
        int order = 2;
        float a[] = {0, 2, -1};
        float a1[3][17] = {0.f};

        a1[0][0] = 1.;
        for (int o = 0; o < order; o++) {
            for (int i = 1; i <= delay_factor; i++) {
                for (int j = 1; j <= i; j++) {
                    if (j + o <= order) {
                        a1[o][i] += a1[0][i - j] * a[j + o];
                    }
                }
            }
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
                expected[x][y] = float16_to_float(img(x, y)) +
                                 (x > 0 ? expected[x - 1][y] * a : 0.f) +
                                 (x > 1 ? expected[x - 2][y] * b : 0.f);
                if (fabs(expected[x][y] - output(x % 16, x / 16, y)) > 0.001f) {
                    std::cerr << "Error at (" << x << ", " << y << "): "
                              << std::fixed << std::setprecision(10)
                              << output(x % 16, x / 16, y) << " != " << expected[x][y] << "\n";
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

    const std::vector<float> scales = {0.07, 0.12, 0.25, 0.75, 0.9, 0.99};  //, 1.01, 1.1, 1.5, 2., 2.5, 4};
    std::string benchmark_name = BENCHMARK_NAME;

    std::cout << "Running " << benchmark_name << " with:" << std::endl;
    std::cout << "  Image size: " << N << "x" << M << std::endl;
    std::cout << "  Schedule: " << SCHEDULE << std::endl;

    for (const float scale : scales) {
        std::cout << "  scale: " << scale << std::endl;

        const int OM = M * scale;
        const int ON = N * scale;

        // Make a pinwheel so that the quality of the resample is apparent
        Buffer<uint16_t, 3> img(M, N, C);
        int R = std::min(M, N) / 2 - 10;
        FOR(c, C) {
            FOR(y, N) {
                double dy = y - N / 2 + 0.5 / scale;
                FOR(x, M) {
                    double dx = x - M / 2 + 0.5 / scale;
                    if (dx * dx + dy * dy > R * R) {
                        img(x, y, c) = float_to_float16(0.5f);
                    } else {
                        double theta = atan2(dy, dx);
                        bool white = ((int)((theta / M_PI + 1) * 300 + 0.5)) & 1;
                        img(x, y, c) = white ? float_to_float16(1.0f) : float_to_float16(0.f);
                    }
                }
            }
        }

        // smallest multiple of 32 that is greater than OM and ON.
        const int OM_realized = (OM + 31) & ~31;
        const int ON_realized = (ON + 31) & ~31;
        Buffer<uint16_t, 3> output(OM_realized, ON_realized, C);

        img.raw_buffer()->type = halide_type_t(halide_type_float, 16);
        output.raw_buffer()->type = halide_type_t(halide_type_float, 16);

        auto resize = scale > 1.0f ? resize_up : resize_down;

        auto time = benchmark(5, 5, [&]() {
            resize(img.raw_buffer(), scale, output.raw_buffer());
            output.device_sync();
        });

        if (output.has_device_allocation()) {
            output.copy_to_host();
        }
        output.device_sync();

        Buffer<uint8_t, 3> output_8(OM, ON, C);
        FOR(c, C) {
            FOR(y, ON) {
                FOR(x, OM) {
                    float f = float16_to_float(output(x, y, c));
                    output_8(x, y, c) = (uint8_t)(f * 255.999f);
                }
            }
        }
        Halide::Tools::save_image(output_8, "pinwheel_" + std::to_string(scale) + ".png");

        std::cout << "Runtime: " << std::fixed << std::setprecision(9) << time << "\n";

        if (VERIFY_OUTPUT) {
            uint16_t *expected = new uint16_t[OM * ON * C];
            resize_sim((uint16_t *)img.raw_buffer()->host, expected, M, N, C, scale);

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
            FOR(c, C) {
                if (!success) break;
                FOR(y, ON) {
                    if (!success) break;
                    FOR(x, OM) {
                        float exp = float16_to_float(expected[c * OM * ON + y * OM + x]);
                        float out = float16_to_float(output(x, y, c));
                        if (fabs(exp - out) > 0.01f) {
                            std::cerr << "Error at (" << x << ", " << y << ", " << c << "): "
                                      << std::fixed << std::setprecision(10)
                                      << out << " != " << exp << "\n";
                            success = false;
                            break;
                        }
                    }
                }
            }

            if (success) {
                std::cout << "Outputs match!\n";
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

constexpr int lanczos_lobes = 3;

float lanczos(float x) {
    float value = sinc(x) * sinc(x / lanczos_lobes);
    value = x == 0.0f ? 1.0f : value;                                  // Take care of singularity at zero
    value = (x > lanczos_lobes || x < -lanczos_lobes) ? 0.0f : value;  // Clamp to zero out of bounds
    return value;
}

float clamp(float x, float l, float h) {
    return std::min(std::max(x, l), h);
}

void resize_sim(uint16_t *input, uint16_t *output, int m, int n, int C, float scale_factor) {
    const int taps = 2 * lanczos_lobes;
    bool upsample = scale_factor > 1.0f;

    float inverse_scale_factor = 1.0f / scale_factor;

    float kernel_scaling = upsample ? 1.0f : scale_factor;
    float inverse_kernel_scaling = upsample ? 1.0f : inverse_scale_factor;

    float kernel_radius = 0.5f * taps * inverse_kernel_scaling;

    int kernel_taps = int(ceil(taps * inverse_kernel_scaling));

    int resized_m = int(m * scale_factor);
    int resized_n = int(n * scale_factor);

    float *kernel_x = new float[kernel_taps * resized_m];
    float *kernel_y = new float[kernel_taps * resized_n];

    FOR(x, resized_m) {
        float sum = 0.f;
        FOR(k, kernel_taps) {
            float sourcex = (x + 0.5f) * inverse_scale_factor - 0.5f;
            int beginx = int(ceil(sourcex - kernel_radius));
            // Compared to the original app, don't need to +1 here because max = min + extent - 1
            beginx = clamp(beginx, 0, m - kernel_taps);
            kernel_x[x * kernel_taps + k] = lanczos((k + beginx - sourcex) * kernel_scaling);
            sum += kernel_x[x * kernel_taps + k];
        }
        FOR(k, kernel_taps) {
            kernel_x[x * kernel_taps + k] /= sum;
        }
    }

    FOR(y, resized_n) {
        float sum = 0.f;
        FOR(k, kernel_taps) {
            float sourcey = (y + 0.5f) * inverse_scale_factor - 0.5f;
            int beginy = int(ceil(sourcey - kernel_radius));
            beginy = clamp(beginy, 0, n - kernel_taps);
            kernel_y[y * kernel_taps + k] = lanczos((k + beginy - sourcey) * kernel_scaling);
            sum += kernel_y[y * kernel_taps + k];
        }
        FOR(k, kernel_taps) {
            kernel_y[y * kernel_taps + k] /= sum;
        }
    }

    float *resized_y = new float[m * resized_n * C];

    FOR(c, C) {
        FOR(y, resized_n) {
            float sourcey = (y + 0.5f) * inverse_scale_factor - 0.5f;
            int beginy = int(ceil(sourcey - kernel_radius));
            beginy = clamp(beginy, 0, n - kernel_taps);
            FOR(x, m) {
                resized_y[c * resized_n * m + y * m + x] = 0.f;
                FOR(r, kernel_taps) {
                    resized_y[c * resized_n * m + y * m + x] +=
                        kernel_y[y * kernel_taps + r] *
                        float16_to_float(input[c * n * m + (beginy + r) * m + x]);
                }
            }
        }
    }

    FOR(c, C) {
        FOR(y, resized_n) {
            FOR(x, resized_m) {
                float out = 0.f;
                float sourcex = (x + 0.5f) * inverse_scale_factor - 0.5f;
                int beginx = int(ceil(sourcex - kernel_radius));
                beginx = clamp(beginx, 0, m - kernel_taps);
                FOR(r, kernel_taps) {
                    out +=
                        kernel_x[x * kernel_taps + r] *
                        resized_y[c * resized_n * m + y * m + r + beginx];
                }
                output[c * resized_n * resized_m + y * resized_m + x] =
                    float_to_float16(clamp(out, 0.f, 1.f));
            }
        }
    }

    delete[] kernel_x;
    delete[] kernel_y;
    delete[] resized_y;
}
