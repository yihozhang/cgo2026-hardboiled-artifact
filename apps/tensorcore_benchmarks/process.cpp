#include "HalideRuntime.h"                                            
#include "HalideRuntimeCuda.h"                                        
#include "HalideBuffer.h"                                             
#include "halide_benchmark.h"                  
                                                                      
#include <iostream>                                                   
#include <cstdlib>  // for rand()                                     
#include <iomanip>  // for std::fixed and std::setprecision
                                                                      
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
            image(x, y) = float_to_float16(1); //uint16_t(rand() % 20);
        }
    }

    // Create output buffer
    Buffer<float> output(imgW - kSize, imgH - kSize);

#if defined(RUN_conv1d)
    // Create kernel buffer                                           
    Buffer<uint16_t> kernel(kSize);                                   
    for (int i = 0; i < kSize; i++) {
        kernel(i) = float_to_float16(1);
    }

    image.raw_buffer()->type = halide_type_t(halide_type_float, 16);
    kernel.raw_buffer()->type = halide_type_t(halide_type_float, 16);
    
    // Call the generated function
    auto time = benchmark(5, 5, [&]() {
        conv1d(kernel.raw_buffer(), image.raw_buffer(), output.raw_buffer());
        output.device_sync();
    });
    
    if (output.has_device_allocation()) {
        output.copy_to_host();
    }

    std::cout << "Runtime: " << time << "\n";

    // Verify results
    if (VERIFY_OUTPUT) {
        bool success = true;
        for (int y = 0; y < imgH; y++) {
            for (int x = 0; x < imgW - kSize; x++) {
                float expected = 0.0f;
                for (int k = 0; k < kSize; k++) {
                    expected += halide_float16_bits_to_float(kernel(k)) * halide_float16_bits_to_float(image(x + k, y));
                }
                if (fabs(expected - output(x, y)) > 0.1f) {
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
#elif defined(RUN_conv2d)
    // Create kernel buffer                                           
    Buffer<uint16_t> kernel(kSize, kSize);                                   
    for (int i = 0; i < kSize; i++) {
        for (int j = 0; j < kSize; j++) {
            kernel(i, j) = float_to_float16(1);
        }
    }
    
    image.raw_buffer()->type = halide_type_t(halide_type_float, 16);
    kernel.raw_buffer()->type = halide_type_t(halide_type_float, 16);

    // Call the generated function
    auto time = benchmark(5, 5, [&]() {   
        conv2d(kernel.raw_buffer(), image.raw_buffer(), output.raw_buffer());
        output.device_sync();
    });
    
    if (output.has_device_allocation()) {
        output.copy_to_host();
    }

    std::cout << "Runtime: " << time << "\n";

    // Verify results
    if (VERIFY_OUTPUT) {
        bool success = true;
        for (int y = 0; y < imgH - kSize; y++) {
            for (int x = 0; x < imgW - kSize; x++) {
                float expected = 0.0f;
                for (int ky = 0; ky < kSize; ky++) {
                    for (int kx = 0; kx < kSize; kx++) {
                        expected += halide_float16_bits_to_float(kernel(kx, ky)) * halide_float16_bits_to_float(image(x + kx, y + ky));
                    }
                }
                if (fabs(expected - output(x, y)) > 0.001f) {
                    std::cerr << "Error at (" << x << ", " << y << "): "
                              << std::fixed << std::setprecision(10) 
                              << output(x, y) << " != " << expected << "\n";
                    success = false;
                    break;
                }
            }
        }

        for (int y = 0; y < 16; y++) {
            for (int x = 0; x < 1; x++) {
                float expected = 0.0f;
                for (int ky = 0; ky < kSize; ky++) {
                    for (int kx = 0; kx < kSize; kx++) {
                        expected += halide_float16_bits_to_float(kernel(kx, ky)) * halide_float16_bits_to_float(image(x + kx, y + ky));
                    }
                }
                std::cout << "(" << x << ", " << y << ") " 
                         << std::fixed << std::setprecision(10) 
                         << output(x, y) << " " << expected << "\n";
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