#include "Halide.h"
#include <stdio.h>

#include "common.h"

using namespace Halide;

class Upsample : public Halide::Generator<Upsample> {
public:
    // Generator Params
    GeneratorParam<Schedule> gpu_schedule{
        "gpu_schedule", Schedule::CUDA,         //
        {                                       //
         {"cuda_only", Schedule::CUDA},         //
         {"tensorcore", Schedule::TensorCore}}  //
    };

    GeneratorParam<int> kSize{"kSize", 16};
    GeneratorParam<int> imgRow{"imgRow", 4096};
    GeneratorParam<int> imgCol{"imgCol", 4096};

    // Inputs
    Input<Buffer<float16_t>> kernel{"kernel", 2};
    Input<Buffer<float16_t>> image{"image", 2};

    // Output
    Output<Buffer<float>> output{"output", 2};

    void generate() {
        Var x{"x"}, y{"y"}, dx{"dx"}, dy{"dy"};

        RDom rk(0, kSize / 2, 0, kSize / 2, "rk");

        // Possibly the cleanest way to treat upsampling in signal processing
        // terms is as interleaving the outputs of a multi-phase filter. We're
        // going to assume the kernel is non-separable (e.g. a Bessel function,
        // which is optimal in some sense).

        // First rearrange the kernel as appropriate to extract each phase
        Func kernel_phases{"kernel_phases"};
        kernel_phases(x, y, dx, dy) = kernel(2 * x + dx, 2 * y + dy);

        Func conv{"conv"};
        conv(x, y, dx, dy) = cast<float>(0);
        conv(x, y, dx, dy) +=
            cast<float>(kernel_phases(rk.x, rk.y, dx, dy)) *
            cast<float>(image(x + rk.x, y + rk.y));

        output(x, y) = conv(x / 2, y / 2, x % 2, y % 2);

        // TODO: Important to clarify how this is different from the conv example

        kernel_phases.compute_root();

        // The output starts at zero and is even-sized
        output.dim(0).set_bounds(0, (output.dim(0).extent() / 2) * 2);
        output.dim(1).set_bounds(0, (output.dim(1).extent() / 2) * 2);

        if (gpu_schedule == Schedule::CUDA) {
            /*---------------------------------*
            |  Tunables                       |
            *---------------------------------*/
            const int blockTileX = 32;
            const int blockTileY = 8;
            const int threadTileX = 2;
            const int threadTileY = 2;
            const int reductionTileX = 1;
            const int reductionTileY = 1;

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
            output.split(y, ty, tyi, 2 * threadTileY)
                .split(x, tx, txi, 2 * threadTileX)
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
                .reorder({dx, dy, txi, tyi, tx, ty})
                .vectorize(txi)
                .unroll(tyi)
                .unroll(dx)
                .unroll(dy);

            /*------------------------------------------------------------------*
            |  3.  Schedule the reduction of conv                              |
            *------------------------------------------------------------------*/
            conv.update()
                .split(y, ty, tyi, threadTileY)
                .split(x, tx, txi, threadTileX)
                .split(rk.y, rkyo, rkyi, reductionTileY)
                .split(rk.x, rkxo, rkxi, reductionTileX)
                .reorder({dx, dy, rkxi, txi, tyi, rkyi, rkxo, rkyo, tx, ty})
                .unroll(rkxi)
                .unroll(rkyi)
                .unroll(dx)
                .unroll(dy)
                .vectorize(txi)
                .unroll(tyi);
        } else if (gpu_schedule == Schedule::TensorCore) {
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

            output.split(y, by, mmy, 2 * blockTileY)
                .split(x, bx, mmx, 2 * blockTileX)
                .gpu_blocks(bx, by)
                .gpu_threads(mmx, mmy)
                .tile(mmx, mmy, mmx, mmy, mmxi, mmyi, 16, 2)
                .reorder({mmxi, mmyi, mmx, mmy, bx, by})
                .unroll(mmxi)
                .unroll(mmyi);
            /*
                .split(mmy, mmy, mmyi, 2 * wmmaTileY)
                .split(mmx, mmx, mmxi, 2 * wmmaTileX)
                .reorder({mmxi, mmyi, mmx, mmy, bx, by})
                .unroll(mmxi, 2)
                .unroll(mmyi, 2)
                .vectorize(mmxi)
                .vectorize(mmyi);
            */

            conv
                .in()
                .compute_at(output, bx)
                .split(y, mmy, mmyi, wmmaTileY)
                .split(x, mmx, mmxi, wmmaTileX)
                .reorder({dx, dy, mmxi, mmyi, mmx, mmy})
                .unroll(dx)
                .unroll(dy)
                .vectorize(mmxi)
                .vectorize(mmyi);

            conv.compute_at(output, bx)
                .store_in(MemoryType::WMMAAccumulator)
                .split(y, mmy, mmyi, wmmaTileY)
                .split(x, mmx, mmxi, wmmaTileX)
                .vectorize(mmxi)
                .vectorize(mmyi)
                .unroll(dx)
                .unroll(dy);

            conv.update()
                .split(y, mmy, mmyi, wmmaTileY)
                .split(x, mmx, mmxi, wmmaTileX)
                .split(rk.x, rkxo, rkxi, reductionTileX)
                .split(rk.y, rkyo, rkyi, reductionTileY)
                .reorder({dx, dy, rkxi, mmxi, mmyi, rkyi, rkxo, rkyo, mmx, mmy})
                .atomic()
                .vectorize(mmxi)
                .vectorize(mmyi)
                .vectorize(rkxi)
                .unroll(rkyi)
                .unroll(rkxo)
                .unroll(rkyo)
                .unroll(dx)
                .unroll(dy);  // TODO: vectorize dx and dy and have a smaller kernel?
        }
    }

private:
};

HALIDE_REGISTER_GENERATOR(Upsample, upsample)
