#include "Halide.h"
#include <stdio.h>

#include "common.h"

using namespace Halide;

class Convolution1D : public Halide::Generator<Convolution1D> {
public:
    // Generator Params
    GeneratorParam<Schedule> gpu_schedule{"gpu_schedule", Schedule::CUDA, {{"cudaonly", Schedule::CUDA}, {"tensorcore", Schedule::TensorCore}}};

    GeneratorParam<int> kSize{"kSize", 128};

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
                .vectorize(txi)
                .vectorize(tyi)
                .atomic();

        } else if (gpu_schedule == Schedule::TensorCore) {
            /*---------------------------------*
            |  Tunables                       |
            *---------------------------------*/
            const int blockTileX = 256;
            const int blockTileY = 16;

            // We compute 256 contiguous elements
            // as a 32x8 matrix
            const int wmmaTileX = 256;
            const int wmmaTileY = 1;

            const int unrollY = 2;

            const int reductionTileX = 8;

            /*---------------------------------*
            |  Vars / RVars                   |
            *---------------------------------*/
            Var by("by"), mmy("mmy"), mmyi("mmyi");
            Var bx("bx"), mmx("mmx"), mmxi("mmxi");
            Var bxi("bxi"), byi("byi");
            RVar rkxo("rkxo"), rkxi("rkxi");

            output.split(y, by, mmy, blockTileY)
                .split(x, bx, mmx, blockTileX)
                .gpu_blocks(bx, by)
                .split(mmy, mmy, mmyi, wmmaTileY * unrollY)
                .split(mmx, mmx, mmxi, wmmaTileX)
                .reorder({mmxi, mmyi, mmx, mmy, bx, by})
                .vectorize(mmxi, wmmaTileX)
                .vectorize(mmyi, wmmaTileY)
                .unroll(mmyi)
                .unroll(mmxi);

            conv.compute_at(output, mmx)
                .store_in(MemoryType::WMMAAccumulator)
                .split(y, mmy, mmyi, wmmaTileY)
                .split(x, mmx, mmxi, wmmaTileX)
                .reorder({mmxi, mmyi, mmx, mmy})
                .unroll(mmx)
                .unroll(mmy)
                .vectorize(mmxi)
                .vectorize(mmyi);

            conv.update()
                .split(y, mmy, mmyi, wmmaTileY)
                .split(x, mmx, mmxi, wmmaTileX)
                .split(rk.x, rkxo, rkxi, reductionTileX)
                .reorder({rkxi, mmxi, mmyi, mmx, mmy, rkxo})
                .unroll(mmx)
                .unroll(mmy)
                .atomic()
                .vectorize(mmxi)
                .vectorize(mmyi)
                .vectorize(rkxi)
                .unroll(rkxo);
        }
    }

private:
    Var x{"x"}, y{"y"};
    RDom rk;
    Func conv{"conv"};
};

HALIDE_REGISTER_GENERATOR(Convolution1D, conv1d)
