#include "Halide.h"
#include <stdio.h>

#include "common.h"

using namespace Halide;

class Convolution2D : public Halide::Generator<Convolution2D> {
public:
    // Generator Params
    GeneratorParam<Schedule> gpu_schedule{"gpu_schedule", Schedule::CUDA, {
        {"cuda_only", Schedule::CUDA},
        {"tensorcore", Schedule::TensorCore}
    }};

    GeneratorParam<int> kSize{"kSize", 16};
    GeneratorParam<int> imgRow{"imgRow", 4096};
    GeneratorParam<int> imgCol{"imgCol", 4096};

    // Inputs
    Input<Buffer<float16_t>> kernel{"kernel", 2};
    Input<Buffer<float16_t>> image{"image", 2};
    
    // Output
    Output<Buffer<float>> output{"output", 2};

    void generate() {
        rk = RDom(0, kSize, 0, kSize, "rk");

        conv(x, y) = cast<float>(0);
        conv(x, y) += cast<float>(kernel(rk.x, rk.y)) * cast<float>(image(x + rk.x, y + rk.y));
        
        output(x, y) = conv(x, y);
    }

    void schedule() {
        if (gpu_schedule == Schedule::CUDA) {
            /*---------------------------------*
            |  Tunables                       |
            *---------------------------------*/
            const int blockTileX = 16;
            const int blockTileY = 16;
            const int threadTileX = 8;
            const int threadTileY = 2;
            const int reductionTileX = 8;
            const int reductionTileY = 8;

            /*---------------------------------*
            |  Vars / RVars                   |
            *---------------------------------*/
            Var by("by"), ty("ty"), tyi("tyi");
            Var bx("bx"), tx("tx"), txi("txi");
            RVar rkxo("rkxo"), rkxi("rkxi");
            RVar rkyo("rkyo"), rkyi("rkyi");
            

            /*------------------------------------------------------------------*
            |  1.  Scheduling the kernel that computes the output. Define GPU   |
            |      blocks and thread tiling.                                    |
            *------------------------------------------------------------------*/
            output.split(y, ty, tyi, threadTileY)
                  .split(x, tx, txi, threadTileX)
                  .split(ty, by, ty, blockTileY)
                  .split(tx, bx, tx, blockTileX)
                  .reorder({txi, tyi, tx, ty, bx, by})
                  .gpu_blocks(bx, by)
                  .gpu_threads(tx, ty)
                  .unroll(txi)
                  .unroll(tyi);

            /*------------------------------------------------------------------*
            |  2.  Scheduling the pure definition of conv                      |
            *------------------------------------------------------------------*/
            conv.compute_at(output, tx)
                .split(y, ty, tyi, threadTileY)
                .split(x, tx, txi, threadTileX)
                .reorder(txi, tyi, tx, ty)
                .vectorize(txi)
                .vectorize(tyi);
            
            /*------------------------------------------------------------------*
            |  3.  Schedule the reduction of conv                              |
            *------------------------------------------------------------------*/
            conv.update()
                .split(y, ty, tyi, threadTileY)
                .split(x, tx, txi, threadTileX)
                .split(rk.y, rkyo, rkyi, reductionTileY)
                .split(rk.x, rkxo, rkxi, reductionTileX)
                .reorder({rkxi, txi, tyi, rkyi, rkxo, rkyo, tx, ty})
                .unroll(rkxi)
                .unroll(rkyi)
                .vectorize(txi)
                .vectorize(tyi)
                .atomic();
        }
        else if (gpu_schedule == Schedule::TensorCore) {
            /*---------------------------------*
            |  Tunables                       |
            *---------------------------------*/
            const int blockTileX = 256;
            const int blockTileY = 1;

            // We compute 256 contiguous elements 
            // as a 32x8 matrix
            const int wmmaTileX = 256;
            const int wmmaTileY = 1;

            const int reductionTileX = 8;
            const int reductionTileY = 1;

            /*---------------------------------*
            |  Vars / RVars                   |
            *---------------------------------*/
            Var by("by"), mmy("mmy"), mmyi("mmyi");
            Var bx("bx"), mmx("mmx"), mmxi("mmxi");
            RVar rkxo("rkxo"), rkxi("rkxi");
            RVar rkyo("rkyo"), rkyi("rkyi");
            
            output.split(y, by, mmy, blockTileY)
                  .split(x, bx, mmx, blockTileX)
                  .gpu_blocks(bx, by)
                  .split(mmy, mmy, mmyi, wmmaTileY)
                  .split(mmx, mmx, mmxi, wmmaTileX)
                  .reorder({mmxi, mmyi, mmx, mmy, bx, by})
                  .vectorize(mmxi)
                  .vectorize(mmyi);

            conv.compute_at(output, mmx)
                .store_in(MemoryType::WMMAAccumulator)
                .split(y, mmy, mmyi, wmmaTileY)
                .split(x, mmx, mmxi, wmmaTileX)
                .vectorize(mmxi)
                .vectorize(mmyi);

            conv.update()
                .split(y, mmy, mmyi, wmmaTileY)
                .split(x, mmx, mmxi, wmmaTileX)
                .split(rk.x, rkxo, rkxi, reductionTileX)
                .split(rk.y, rkyo, rkyi, reductionTileY)
                .reorder({rkxi, mmxi, mmyi, rkyi, rkxo, rkyo, mmx, mmy})            
                .atomic()
                .vectorize(mmxi)
                .vectorize(mmyi)
                .vectorize(rkxi)
                .unroll(rkyi)
                .unroll(rkxo)
                .unroll(rkyo);
        }
    }

private:
    Var x{"x"}, y{"y"};
    RDom rk;
    Func conv{"conv"};
};

HALIDE_REGISTER_GENERATOR(Convolution2D, conv2d)