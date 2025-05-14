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

    GeneratorParam<int> kSize{"kSize", 128};
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
        if (true) {
            Var xi("xi"), yi("yi");
            RVar rxi("rxi"), ryi("ryi");
            Var mmxi("mmxi"),
                mmyi("mmyi");
            RVar mmri("mmri");
            Var xy("xy"), xyi("xyi");

            int tile_x = 4;
            int tile_y = 4;
            int tile_rx = 4;
            int tile_ry = 4;

            // update
            conv.compute_at(output, xi)
                .update()
                .tile(x, y, mmxi, mmyi, tile_x, tile_y)
                .tile(r.x, r.y, rxi, ryi, tile_rx, tile_ry)
                .reorder({ rxi, ryi, mmxi, mmyi, r.x, r.y, x, y})
                .unroll(rxi)
                .unroll(ryi)
                ;
            conv
                .tile(x, y, mmxi, mmyi, tile_x, tile_y)
                .reorder(mmxi, mmyi, x, y)
                .unroll(mmxi)
                .unroll(mmyi);
            output
                .tile(x, y, mmxi, mmyi, tile_x, tile_y)
                .gpu_tile(x, y, xi, yi, 16, 16)
                .reorder({mmxi, mmyi, xi, yi, x, y})
                .gpu_blocks(x, y)
                .gpu_threads(xi, yi)
                .unroll(mmxi)
                .unroll(mmyi);
        }
        else if (gpu_schedule == Schedule::CUDA) {
            /*---------------------------------*
            |  Tunables                       |
            *---------------------------------*/
            const int blockTileX = 128;
            const int blockTileY = 8;
            const int threadTileX = 4;
            const int threadTileY = 2;
            const int reductionTileX = 8;
            const int reductionTileY = 8;
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
                .split(rk.y, rkyo, rkyi, reductionTileY)
                .split(rk.x, rkxo, rkxi, reductionTileX)
                .reorder({rkxi, rkyi, txi, tyi, rkxo, rkyo, tx, ty})
                .unroll(rkxi)
                .unroll(rkyi)
                .atomic();
            
        }
        else if (gpu_schedule == Schedule::TensorCore) {
            /*---------------------------------*
            |  Vars / RVars                   |
            *---------------------------------*/
            Var by("by"), ty("ty"), tyi("tyi");
            Var bx("bx"), tx("tx"), txi("txi");
            RVar rkxo("rkxo"), rkxi("rkxi");

            output.split(x, bx, tx, 8)
                  .split(y, by, ty, 64)
                  .split(tx, tx, txi, 8)
                  .split(ty, ty, tyi, 32)
                  .gpu_blocks(bx, by)
                  .reorder({txi, tyi, tx, ty, bx, by})
                  .unroll(tx)
                  .unroll(ty)
                  .vectorize(txi)
                  .vectorize(tyi);

            conv.compute_at(output, bx)
                .store_in(MemoryType::WMMAAccumulator)
                .split(x, tx, txi, 8)
                .split(y, ty, tyi, 64)
                .atomic()
                .vectorize(txi)
                .vectorize(tyi);

            conv.update()
                .split(x, tx, txi, 8)
                .split(y, ty, tyi, 32)
                .split(rk.x, rkxo, rkxi, 8)
                // I cannot unroll rro, since the temporary buffer refers to
                // rro but is lifted to the host, where rro is not available.
                // .unroll(rkxo)
                .unroll(ty)
                .reorder({rkxi, txi, tyi, tx, ty, rkxo})                
                .atomic()
                .vectorize(rkxi)
                .vectorize(txi)
                .vectorize(tyi);

            image.in()
                 .compute_at(conv, tx)
                 .store_in(MemoryType::WMMAA)
                 .vectorize(_0)
                 .vectorize(_1);
        }
    }

private:
    Var x{"x"}, y{"y"};
    RDom rk;
    Func conv{"conv"};
};

HALIDE_REGISTER_GENERATOR(Convolution2D, conv2d)