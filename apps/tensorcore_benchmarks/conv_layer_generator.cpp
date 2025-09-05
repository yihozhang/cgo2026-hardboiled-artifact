#include "Halide.h"

#include "common.h"

namespace {

using namespace Halide;

class ConvolutionLayer : public Halide::Generator<ConvolutionLayer> {
public:
    // Generator Params
    GeneratorParam<Schedule> gpu_schedule{"gpu_schedule", Schedule::CUDA, {
        {"cuda_only", Schedule::CUDA},
        {"tensorcore", Schedule::TensorCore}
    }};

    GeneratorParam<int> N{"N", 128};
    GeneratorParam<int> H{"H", 64};
    GeneratorParam<int> W{"W", 64};
    GeneratorParam<int> C{"C", 128};
    GeneratorParam<int> kSize{"kSize", 3};

    // Inputs
    Input<Buffer<float16_t, 4>> input{"input"};
    Input<Buffer<float16_t, 4>> filter{"filter"};
    Input<Buffer<float, 1>> bias{"bias"};

    // Output
    Output<Buffer<float, 4>> output{"output"};

    void generate() {
        rk = RDom(0, C, 0, kSize, 0, kSize);

        conv(c, x, y, n) = bias(c);
        conv(c, x, y, n) += cast<float>(filter(c, rk.y, rk.z, rk.x)) * cast<float>(input(rk.x, x + rk.y, y + rk.z, n));

        relu(c, x, y, n) = max(0, conv(c, x, y, n));

        output(c, x, y, n) = relu(c, x, y, n);
    }

    void schedule() {
        int _C = C, _H = H, _W = W, _N = N, _kSize = kSize;

        output.dim(0).set_bounds(0, _C).set_stride(1);
        output.dim(1).set_bounds(0, _W).set_stride(_C);
        output.dim(2).set_bounds(0, _H).set_stride(_C * _W);
        output.dim(3).set_bounds(0, _N).set_stride(_C * _H * _W);

        input.dim(0).set_bounds(0, _C).set_stride(1);
        input.dim(1).set_bounds(0, (_W + _kSize)).set_stride(_C);
        input.dim(2).set_bounds(0, (_H + _kSize)).set_stride(_C * (_W + _kSize));
        input.dim(3).set_bounds(0, _N).set_stride(_C * (_W + _kSize) * (_H + _kSize));

        filter.dim(0).set_bounds(0, _C).set_stride(1);
        filter.dim(1).set_bounds(0, _kSize).set_stride(_C);
        filter.dim(2).set_bounds(0, _kSize).set_stride(_C * _kSize);
        filter.dim(3).set_bounds(0, _C).set_stride(_C * _kSize * _kSize);

        bias.dim(0).set_bounds(0, _C).set_stride(1);

        if (gpu_schedule == Schedule::CUDA) {
            // GPU schedule, tuned for a GTX 980. Seems to be good on
            // an RTX 2060 too (About 90% peak flops on both cards).

            // 1.87 ms on an RTX 2060. According to NVIDIA Nsight
            // Compute we're at 91.5% utilization of the FMA units

            // 2.41 ms on a GTX 980. According to nvprof this is about
            // 88% of peak flops.

            // We use cuda-specific scheduling directives (gpu_lanes),
            // so this is not a general GPGPU schedule.

            Var ni, no, xi, xo, yi, yo, ci, co, t;
            RVar rxo, rxi, rxii;

            output.compute_root()
                .split(x, xo, xi, 4)
                .split(y, yo, yi, 4)
                .split(c, co, ci, 32)
                .reorder(xi, yi, ci, xo, yo, co, n)
                .gpu_lanes(ci)
                .unroll(xi)
                .unroll(yi)
                .fuse(co, n, t)
                .gpu_blocks(xo, yo, t);

            conv.compute_at(output, xo)
                .store_in(MemoryType::Register)
                .gpu_lanes(c)
                .unroll(x)
                .unroll(y)
                .update()
                .split(rk.x, rxo, rxi, 16)
                .split(rxi, rxi, rxii, 2)
                .reorder(c, rxii, x, y, rk.y, rk.z, rxi, rxo)
                .gpu_lanes(c)
                .unroll(x)
                .unroll(y)
                .unroll(rk.y)
                .unroll(rk.z)
                .unroll(rxii);

            input.in()
                .compute_at(conv, rxo)
                .vectorize(_0, 2)
                .split(_1, xo, xi, 4)
                .fuse(_0, xi, t)
                .gpu_lanes(t)
                .unroll(xo)
                .unroll(_2);
        }
        else if (gpu_schedule == Schedule::TensorCore) {
            Var xi("xi"), xii("xii"), yi("yi"), yii("yii");
            Var co("co"), ci("ci");
            RVar rkxo("rkxo"), rkxi("rkxi"), rtile("rtile");
            const int tile_w = 64, tile_h = 4, tile_c = 16, tile_r = 16;

            output
                .split(y, y, yi, tile_h)
                .split(x, x, xi, tile_w)
                .split(xi, xi, xii, 16)
                .split(c, co, ci, tile_c)
                .reorder(ci, xii, xi, yi, co, x, y, n)
                .gpu_blocks(x, y, n)
                .vectorize(ci)
                .vectorize(xii)
                .unroll(xi, 4)
                .unroll(yi, 4)
                .unroll(co);
            
            conv.compute_at(output, co)
                .store_in(MemoryType::WMMAAccumulator)
                .split(x, x, xi, 16)
                .vectorize(c)
                .vectorize(xi)
                .unroll(x, 4)
                .unroll(y, 4);
  
            conv.update()
                .split(x, x, xi, 16)
                .split(rk.x, rkxo, rkxi, tile_r)
                .reorder(rkxi, c, x, y, rkxo, rk.y, rk.z)
                .atomic()
                .vectorize(c)
                .vectorize(xi)
                .vectorize(rkxi)
                .unroll(x, 4)
                .unroll(y, 4);
        }
    }

private:
    Var x{"x"}, y{"y"}, c{"c"}, n{"n"};

    Func conv{"conv"};
    Func relu{"relu"};
    
    RDom rk;
};

}  // namespace

HALIDE_REGISTER_GENERATOR(ConvolutionLayer, conv_layer)