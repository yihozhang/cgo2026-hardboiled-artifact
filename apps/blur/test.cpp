#include <cmath>
#include <cstdint>
#include <cstdio>
#include <iostream>
#include <thread>
// #include <omp.h>
#include <emmintrin.h>

#include "HalideBuffer.h"
#include "halide_benchmark.h"

using namespace Halide::Runtime;
using namespace Halide::Tools;

double t;

Buffer<uint16_t, 2> blur(Buffer<uint16_t, 2> in) {
    Buffer<uint16_t, 2> tmp(in.width() - 8, in.height());
    Buffer<uint16_t, 2> out(in.width() - 8, in.height() - 2);

    // This version is slower than the Halide vanilla schedule by 2~3x because
    // each indexing into buffer requires bounds check
    t = benchmark(5, 5, [&]() {
        for (int y = 0; y < tmp.height(); y++)
            for (int x = 0; x < tmp.width(); x++)
                tmp(x, y) = (in(x, y) + in(x + 1, y) + in(x + 2, y)) / 3;

        for (int y = 0; y < out.height(); y++)
            for (int x = 0; x < out.width(); x++)
                out(x, y) = (tmp(x, y) + tmp(x, y + 1) + tmp(x, y + 2)) / 3;
    });

    return out;
}

Buffer<uint16_t, 2> blur_fast(Buffer<uint16_t, 2> in) {
    Buffer<uint16_t, 2> out(in.width() - 8, in.height() - 2);
    constexpr int TILE_Y = 32;
    t = benchmark(5, 5, [&]() {
#pragma omp parallel for schedule(dynamic, 2)
        for (int yTile = 0; yTile < out.height(); yTile += TILE_Y) {
            __m128i one_third = _mm_set1_epi16(21846);
            __m128i tmp[(128 / 8) * (TILE_Y + 2)];
            for (int xTile = 0; xTile < out.width(); xTile += 128) {
                __m128i *tmpPtr = tmp;
                for (int y = 0; y < TILE_Y + 2; y++) {
                    const uint16_t *inPtr = &(in(xTile, yTile + y));
                    for (int x = 0; x < 128; x += 8) {
                        __m128i a = _mm_loadu_si128((const __m128i *)(inPtr));
                        __m128i b = _mm_loadu_si128((const __m128i *)(inPtr + 1));
                        __m128i c = _mm_loadu_si128((const __m128i *)(inPtr + 2));
                        __m128i sum = _mm_add_epi16(_mm_add_epi16(a, b), c);
                        __m128i avg = _mm_mulhi_epi16(sum, one_third);
                        _mm_store_si128(tmpPtr++, avg);
                        inPtr += 8;
                    }
                }
                tmpPtr = tmp;
                for (int y = 0; y < TILE_Y; y++) {
                    __m128i *outPtr = (__m128i *)(&(out(xTile, yTile + y)));
                    for (int x = 0; x < 128; x += 8) {
                        __m128i a = _mm_load_si128(tmpPtr + (2 * 128) / 8);
                        __m128i b = _mm_load_si128(tmpPtr + 128 / 8);
                        __m128i c = _mm_load_si128(tmpPtr++);
                        __m128i sum = _mm_add_epi16(_mm_add_epi16(a, b), c);
                        __m128i avg = _mm_mulhi_epi16(sum, one_third);
                        _mm_store_si128(outPtr++, avg);
                    }
                }                    
            }                
        }
    });
    return out;
}

#include "halide_blur.h"

Buffer<uint16_t, 2> blur_halide(Buffer<uint16_t, 2> in) {
    Buffer<uint16_t, 2> out(in.width() - 8, in.height() - 2);
    printf("start halide blur\n");
    t = benchmark(5, 5, [&]() {
        // Compute the same region of the output as blur_fast (i.e., we're
        // still being sloppy with boundary conditions)
        halide_blur(in, out);
        // Sync device execution if any.
        out.device_sync();
    });
    printf("done halide blur\n");

    out.copy_to_host();

    return out;
}

int main(int argc, char **argv) {
    // 8K picture
    const int width =  7688;
    const int height = 4322;

    Buffer<uint16_t, 2> input(width, height);

    for (int y = 0; y < input.height(); y++) {
        for (int x = 0; x < input.width(); x++) {
            input(x, y) = (y + x) & 0xfff;
        }
    }

    Buffer<uint16_t, 2> blurry = blur(input);
    double slow_time = t * 1000;
    printf("slow blur time: %f ms\n", slow_time);
    printf("slow blur throughput: %.1f Mpixels/s\n",
           (double)(blurry.width() * blurry.height()) / (slow_time * 1000));

    Buffer<uint16_t, 2> speedy = blur_fast(input);
    double fast_time = t * 1000;
    printf("fast blur time: %f ms\n", fast_time);
    printf("fast blur throughput: %.1f Mpixels/s\n",
            (double)(speedy.width() * speedy.height()) / (fast_time * 1000));
    
    Buffer<uint16_t, 2> halide = blur_halide(input);
    double halide_time = t * 1000;
    printf("halide blur time: %f ms\n", halide_time);
    printf("halide blur throughput: %.1f Mpixels/s\n",
            (double)(halide.width() * halide.height()) / (halide_time * 1000));
        
    for (int y = 64; y < input.height() - 64; y++) {
        for (int x = 64; x < input.width() - 64; x++) {
            if (blurry(x, y) != speedy(x, y) || blurry(x, y) != halide(x, y)) {
                printf("difference at (%d,%d): %d %d %d\n", x, y, blurry(x, y), speedy(x, y), halide(x, y));
                abort();
            }
        }
    }
        
    printf("Outputs match!\n");
    return 0;
}
