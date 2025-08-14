#include "Halide.h"

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
    GeneratorParam<int> H{"H", 56};
    GeneratorParam<int> W{"W", 56};
    GeneratorParam<int> C{"C", 128};
    GeneratorParam<int> kSize{"kSize", 3};

    // Inputs
    Input<Buffer<float, 4>> input{"input"};
    Input<Buffer<float, 4>> filter{"filter"};
    Input<Buffer<float, 1>> bias{"bias"};

    // Output
    Output<Buffer<float, 4>> output{"output"};

    void generate() {
        rk = RDom(0, C, 0, kSize, 0, kSize);

        conv(c, x, y, n) = bias(c);
        conv(c, x, y, n) += filter(c, rk.y, rk.z, rk.x) * input(rk.x, x + rk.y, y + rk.z, n);

        relu(c, x, y, n) = max(0, conv(c, x, y, n));

        output(c, x, y, n) = relu(c, x, y, n);
    }

    void schedule() {
        /* THE SCHEDULE */

        // MKL JITs code for the specific size and strides, so we'll
        // do the same and ask Halide to compile for this specific
        // size:

        output.dim(0).set_bounds(0, C).set_stride(1);
        output.dim(1).set_bounds(0, W).set_stride(C);
        output.dim(2).set_bounds(0, H).set_stride(C * W);
        output.dim(3).set_bounds(0, N).set_stride(C * H * W);

        input.dim(0).set_bounds(0, C).set_stride(1);
        input.dim(1).set_bounds(0, W).set_stride(C);
        input.dim(2).set_bounds(0, H).set_stride(C * W);
        input.dim(3).set_bounds(0, N).set_stride(C * W * H);

        filter.dim(0).set_bounds(0, C).set_stride(1);
        filter.dim(1).set_bounds(0, 3).set_stride(C);
        filter.dim(2).set_bounds(0, 3).set_stride(C * kSize);
        filter.dim(3).set_bounds(0, C).set_stride(C * kSize * kSize);

        bias.dim(0).set_bounds(0, C).set_stride(1);

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

            conv.compute_at(relu, xo)
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

        } else {
            // todo later
        }
    }

private:
    Var x("x"), y("y"), c("c"), n("n");

    Func conv("conv");
    Func relu("relu");
    
    RDom rk;
};

}  // namespace

HALIDE_REGISTER_GENERATOR(ConvolutionLayer, conv_layer)