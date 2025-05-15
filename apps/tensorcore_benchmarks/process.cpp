#include "HalideRuntime.h"                                            
#include "HalideRuntimeCuda.h"                                        
#include "HalideBuffer.h"                                             
#include "halide_benchmark.h"                  
                                                                      
#include <iostream>                                                   
#include <cstdlib>  // for rand()                                     
                                                                      
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
                                                                      
    // Create kernel buffer                                           
    Buffer<uint16_t> kernel(kSize);                                   
    for (int i = 0; i < kSize; i++) {
        kernel(i) = uint16_t(i);
    }

    // Create image buffer with random values
    Buffer<uint16_t> image(imgW, imgH);
    for (int y = 0; y < imgH; y++) {
        for (int x = 0; x < imgW; x++) {
            image(x, y) = uint16_t(rand() % 100);
        }
    }

    // Create output buffer
    Buffer<float> output(imgW - kSize, imgH);

    // Call the generated function
    auto time = benchmark(5, 5, [&]() {

#if defined(RUN_conv1d)
        conv1d(kernel.raw_buffer(), image.raw_buffer(), output.raw_buffer());
#elif defined(RUN_conv2d)
        conv2d(kernel.raw_buffer(), image.raw_buffer(), output.raw_buffer());
#else
    #error "Unknown benchmark type"
#endif

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
                    expected += bfloat16_to_float(kernel(k)) * bfloat16_to_float(image(x + k, y));
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

    return 0;
}