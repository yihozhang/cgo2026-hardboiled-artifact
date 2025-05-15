#include "Halide.h"
#include <stdio.h>

#include "common.h"

using namespace Halide;

class Convolution1D : public Halide::Generator<Convolution1D> {
public:
    // Generator Params
    GeneratorParam<Schedule> gpu_schedule{"gpu_schedule", Schedule::CUDA, {
        {"cuda_only", Schedule::CUDA},
        {"tensorcore", Schedule::TensorCore}
    }};

    GeneratorParam<int> kSize{"kSize", 128};
    GeneratorParam<int> imgRow{"imgRow", 4096};
    GeneratorParam<int> imgCol{"imgCol", 4096};

    // Inputs
    Input<Buffer<float16_t>> kernel{"kernel", 1};
    Input<Buffer<float16_t>> image{"image", 2};
    
    // Output
    Output<Buffer<float>> output{"output", 2};

    void generate() {
        rk = RDom(0, kSize, "rk");

        conv(x, y) = cast<float>(0);
        conv(x, y) += cast<float>(kernel(rk.x)) * cast<float>(image(x + rk.x, y));
        
        output(x, y) = conv(x, y);
    }

    void schedule() {
        if (gpu_schedule == Schedule::CUDA) {
            /*---------------------------------*
            |  Tunables                       |
            *---------------------------------*/
            const int blockTileX = 128;
            const int blockTileY = 8;
            const int threadTileX = 4;
            const int threadTileY = 2;
            const int reductionTileX = 8;

            /*---------------------------------*
            |  Vars / RVars                   |
            *---------------------------------*/
            Var by("by"), ty("ty"), tyi("tyi");
            Var bx("bx"), tx("tx"), txi("txi");
            RVar rkxo("rkxo"), rkxi("rkxi");
            

            /*------------------------------------------------------------------*
            |  1.  Scheduling the kernel that computes the output. Define GPU   |
            |      blocks and thread tiling.                                    |
            *------------------------------------------------------------------*/
            output.split(y, by, ty, blockTileY)
                  .split(x, bx, tx, blockTileX)
                  .split(ty, ty, tyi, threadTileY)
                  .split(tx, tx, txi, threadTileX)
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
                .atomic()
                .vectorize(txi)
                .vectorize(tyi);

            /*------------------------------------------------------------------*
            |  3.  Schedule the reduction of conv                              |
            *------------------------------------------------------------------*/
            conv.update()
                .split(y, ty, tyi, threadTileY)
                .split(x, tx, txi, threadTileX)
                .split(rk.x, rkxo, rkxi, reductionTileX)
                .reorder({rkxi, txi, tyi, rkxo, tx, ty})
                .unroll(rkxi)
                .atomic();
            
        }
        else if (gpu_schedule == Schedule::TensorCore) {
            /*---------------------------------*
            |  Tunables                       |
            *---------------------------------*/
            const int blockTileX = 8;
            const int blockTileY = 32;
            const int threadTileX = 8;
            const int threadTileY = 32;
            const int reductionTileX = 8;

            /*---------------------------------*
            |  Vars / RVars                   |
            *---------------------------------*/
            Var by("by"), ty("ty"), tyi("tyi");
            Var bx("bx"), tx("tx"), txi("txi");
            RVar rkxo("rkxo"), rkxi("rkxi");

            output.split(y, by, ty, blockTileY)
                  .split(x, bx, tx, blockTileX)
                  .split(tx, tx, txi, threadTileX)
                  .split(ty, ty, tyi, threadTileY)
                  .gpu_blocks(bx, by)
                  .reorder({txi, tyi, tx, ty, bx, by})
                  .unroll(tx)
                  .unroll(ty)
                  .vectorize(txi)
                  .vectorize(tyi);

            conv.compute_at(output, bx)
                .store_in(MemoryType::WMMAAccumulator)
                .split(y, ty, tyi, threadTileY)
                .split(x, tx, txi, threadTileX)
                //.unroll(ty)
                .vectorize(txi)
                .vectorize(tyi);

            conv.update()
                .split(y, ty, tyi, threadTileY)
                .split(x, tx, txi, threadTileX)
                .split(rk.x, rkxo, rkxi, reductionTileX)
                //.unroll(ty)
                .reorder({rkxi, txi, tyi, tx, ty, rkxo})
                .atomic()
                .vectorize(rkxi)
                .vectorize(txi)
                .vectorize(tyi)
                .unroll(rkxo);
        }
    }

private:
    Var x{"x"}, y{"y"};
    RDom rk;
    Func conv{"conv"};
};

HALIDE_REGISTER_GENERATOR(Convolution1D, conv1d)