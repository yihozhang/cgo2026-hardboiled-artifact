#include "Halide.h"
#include <stdio.h>

#include "common.h"
#include "Halide.h"

using namespace Halide;

enum InterpolationType {
    Box,
    Linear,
    Cubic,
    Lanczos
};

Expr kernel_box(Expr x) {
    Expr xx = abs(x);
    return select(xx <= 0.5f, 1.0f, 0.0f);
}

Expr kernel_linear(Expr x) {
    Expr xx = abs(x);
    return select(xx < 1.0f, 1.0f - xx, 0.0f);
}

Expr kernel_cubic(Expr x) {
    Expr xx = abs(x);
    Expr xx2 = xx * xx;
    Expr xx3 = xx2 * xx;
    float a = -0.5f;

    return select(xx < 1.0f, (a + 2.0f) * xx3 - (a + 3.0f) * xx2 + 1,
                  select(xx < 2.0f, a * xx3 - 5 * a * xx2 + 8 * a * xx - 4.0f * a,
                         0.0f));
}

Expr sinc(Expr x) {
    x *= 3.14159265359f;
    return sin(x) / x;
}

Expr kernel_lanczos(Expr x) {
    Expr value = sinc(x) * sinc(x / 3);
    value = select(x == 0.0f, 1.0f, value);        // Take care of singularity at zero
    value = select(x > 3 || x < -3, 0.0f, value);  // Clamp to zero out of bounds
    return value;
}

struct KernelInfo {
    const char *name;
    int taps;
    Expr (*kernel)(Expr);
};

static KernelInfo kernel_info[] = {
    {"box", 1, kernel_box},
    {"linear", 2, kernel_linear},
    {"cubic", 4, kernel_cubic},
    {"lanczos", 6, kernel_lanczos}};

class Resize : public Halide::Generator<Resize> {
public:
    GeneratorParam<InterpolationType> interpolation_type{"interpolation_type", Cubic, {{"box", Box}, {"linear", Linear}, {"cubic", Cubic}, {"lanczos", Lanczos}}};

    // If we statically know whether we're upsampling or downsampling,
    // we can generate different pipelines (we want to reorder the
    // resample in x and in y).
    GeneratorParam<bool> upsample{"upsample", false};
    GeneratorParam<Schedule> gpu_schedule {
        "gpu_schedule", Schedule::CUDA,         //
        {                                       //
         {"cuda_only", Schedule::CUDA},         //
         {"tensorcore", Schedule::TensorCore}}  //
    };


    // We change from void to float for convience
    Input<Buffer<float, 3>> input{"input"};
    Input<float> scale_factor{"scale_factor"};
    Output<Buffer<float, 3>> output{"output"};

    // Common Vars
    Var x{"x"}, y{"y"}, c{"c"}, k{"k"};
    Var z{"z"}, w{"w"};
    Var zo{"zo"}, zi{"zi"};
    Var xi{"xi"}, yi{"yi"};
    Var xo{"xo"}, yo{"yo"};
    RDom r;
    RVar ri{"ri"}, ro{"ro"};
    RVar rxi{"rxi"}, rxo{"rxo"}, ryi{"ryi"}, ryo{"ryo"};
    RDom rx, ry;

    // Intermediate Funcs
    Func as_float{"as_float"},
        resized_x{"resized_x"},
        resized_y{"resized_y"},
        resized_yf16{"resized_yf16"},
        unnormalized_kernel_x{"unnormalized_kernel_x"},
        unnormalized_kernel_y{"unnormalized_kernel_y"},
        kernel_x{"kernel_x"},
        kernel_y{"kernel_y"},
        kernel_sum_x{"kernel_sum_x"},
        kernel_sum_y{"kernel_sum_y"};
    
    Func resized{"resized"};
    Func is_empty_block_x{"is_empty_block_x"}, kernel_blocks_x{"kernel_blocks_x"};
    Func is_empty_block_y{"is_empty_block_y"}, kernel_blocks_y{"kernel_blocks_y"};

    Expr inverse_scale_factor;
    Expr kernel_scaling;
    Expr inverse_kernel_scaling;
    Expr kernel_radius;
    Expr kernel_taps;

    const int block_size = 16;

    // Since we allow an
    // arbitrary scaling factor, the filter coefficients are
    // different for each x and y coordinate. Use strict-float to
    // ensure fast-math doesn't mess up our bounds inference.
    bool strict = false;

    Expr begin_of(Expr x) {
        Expr sourcex = (x + 0.5f) * inverse_scale_factor - 0.5f;
        
        Expr beginx;
        if (strict) {
            beginx = cast<int>(strict_float(ceil(sourcex - kernel_radius)));
        } else {
            beginx = cast<int>(ceil(sourcex - kernel_radius));
        }
        beginx = clamp(beginx, input.dim(0).min(), input.dim(0).max() + 1 - kernel_taps);
        return beginx;
    }

    void generate() {

        // Handle different types by just casting to float
        as_float(x, y, c) = cast<float16_t>(input(x, y, c));

        // For downscaling, widen the interpolation kernel to perform lowpass
        // filtering.

        // Invert the scale factor in a single place and do it
        // strictly, to avoid getting different ratios showing up in
        // different places.
        // Expr inverse_scale_factor = strict_float(1.0f / scale_factor);
        inverse_scale_factor = 1.0f / scale_factor;

        kernel_scaling = upsample ? Expr(1.0f) : scale_factor;
        inverse_kernel_scaling = upsample ? Expr(1.0f) : inverse_scale_factor;

        kernel_radius = 0.5f * kernel_info[interpolation_type].taps * inverse_kernel_scaling;

        kernel_taps = cast<int>(ceil(kernel_info[interpolation_type].taps * inverse_kernel_scaling));

        // source[xy] are the (non-integer) coordinates inside the source image
        Expr sourcex = (x + 0.5f) * inverse_scale_factor - 0.5f;
        Expr sourcey = (y + 0.5f) * inverse_scale_factor - 0.5f;

        Expr beginx = begin_of(x);
        Expr beginy = begin_of(y);
        

        r = {0, kernel_taps, "r"};
        const KernelInfo &info = kernel_info[interpolation_type];

        unnormalized_kernel_x(x, k) = info.kernel((k + beginx - sourcex) * kernel_scaling);
        unnormalized_kernel_y(y, k) = info.kernel((k + beginy - sourcey) * kernel_scaling);

        kernel_sum_x(x) = sum(unnormalized_kernel_x(x, r), "kernel_sum_x");
        kernel_sum_y(y) = sum(unnormalized_kernel_y(y, r), "kernel_sum_y");

        kernel_x(x, k) = cast<float16_t>(unnormalized_kernel_y(x, k) / kernel_sum_x(x));
        kernel_y(y, k) = cast<float16_t>(unnormalized_kernel_y(y, k) / kernel_sum_y(y));

        // Perform separable resizing. The resize in x vectorizes
        // poorly compared to the resize in y, so do it first if we're
        // upsampling, and do it second if we're downsampling.
        if (upsample) {
            resized_x(x, y, c) = sum(kernel_x(x, r) * as_float(r + beginx, y, c), "resized_x");
            resized_y(x, y, c) = sum(kernel_y(y, r) * resized_x(x, r + beginy, c), "resized_y");
            resized = resized_y;
        } else {
            // -------------- original algorithm -------------- 
            // resized_y(x, y, c) = cast<float16_t>(0.f);
            // resized_y(x, y, c) += cast<float16_t>(cast<float>(kernel_y(y, r)) * cast<float>(as_float(x, r + beginy, c)));
            // ------------------------------------------------

            // For a block to be non-empty, intervals [16x, 16x+15] and [begin(16y), begin(16y+15)+taps] need to intersect
            // begin(x) / 16
            is_empty_block_y(x, y) = 
                (x * block_size <= begin_of(block_size * y + block_size - 1) + kernel_taps) && 
                begin_of(block_size * y) <= x * block_size + block_size - 1;

            // for y by 16
            //    start_x = floor(begin(y) / 16)
            //    end_x  = floor(begin(y + 15) / 16) + (ceil(kernel_width / 16) + 1)

            // xi and xo denote rows of the band kernel matrix
            /*-------------------
             |r___|__           |
             |  |r___|__        |
             |     |r___|       |
             |        ...       |
             |                  |
             |                  |
             |                  |
             |                  |
              -------------------
            */
            Expr offset_from_beginy = (xo * block_size + xi) - begin_of(yo * block_size + yi);
            kernel_blocks_y(xi, yi, xo, yo) = 
                select(0 <= offset_from_beginy && offset_from_beginy <= kernel_taps, kernel_y(yo * block_size + yi, offset_from_beginy), 0);
            
            ry = {0, block_size, 0, input.dim(1).extent() / 16, "ry"};
            ry.where(is_empty_block_y(ry.y, yo));
            resized_y(xi, yi, xo, yo, c) = 0.f;
            resized_y(xi, yi, xo, yo, c) += 
                cast<float>(kernel_blocks_y(ry.x, yi, ry.y, yo)) * 
                cast<float>(as_float(xi + xo * block_size, ry.x + ry.y * block_size, c));
            resized_yf16(xi, yi, xo, yo, c) = cast<float16_t>(resized_y(xi, yi, xo, yo, c));

            // -------------- original algorithm --------------
            // resized_x(x, y, c) = 0.f;
            // resized_x(x, y, c) += cast<float>(kernel_x(x, r)) * cast<float>(resized_y(r + beginx, y, c));
            // ------------------------------------------------

            // For a block to be non-empty, intervals [16y, 16y+15] and [begin(16x), begin(16x+15)+taps] need to intersect
            is_empty_block_x(x, y) = 
                (y * block_size <= begin_of(block_size * x + block_size - 1) + kernel_taps) && 
                begin_of(block_size * x) <= y * block_size + block_size - 1;
            

            // yi and yo denote columns of the band kernel matrix
            /*-------------------
             | |                |
             |r|_               |
             |_| |              |
             | |r|_             |
             | |_| |            |
             |   |r|            |
             |   |_|            |
             |      ...         |
              -------------------
            */
            Expr offset_from_beginx = (yo * block_size + yi) - begin_of(xo * block_size + xi);
            kernel_blocks_x(xi, yi, xo, yo) = 
                select(0 <= offset_from_beginx && offset_from_beginx <= kernel_taps, kernel_x(xo * block_size + xi, offset_from_beginx), 0);
            
            rx = {0, block_size, 0, input.dim(0).extent() / 16, "rx"};
            rx.where(is_empty_block_x(xo, rx.y));
            resized_x(xi, yi, xo, yo, c) = 0.f;
            resized_x(xi, yi, xo, yo, c) += 
                cast<float>(kernel_blocks_x(xi, rx.x, xo, rx.y)) * 
                cast<float>(resized_yf16(rx.x, yi, rx.y, yo, c));

            resized(x, y, c) = resized_x(x % block_size, y % block_size, x / block_size, y / block_size, c);
            // resized = resized_x;
        }

        if (input.type().is_float()) {
            output(x, y, c) = clamp(resized(x, y, c), 0.0f, 1.0f);
        } else {
            output(x, y, c) = saturating_cast(input.type(), resized(x, y, c));
        }
    }

    void schedule() {
        // const int vec = natural_vector_size<float>();
        const int vec = 16;

        unnormalized_kernel_x
            .compute_at(kernel_x, x)
            .store_in(MemoryType::Stack)
            .vectorize(x);
        kernel_sum_x
            .compute_at(kernel_x, x)
            .vectorize(x);
        kernel_x
            .compute_root()
            .reorder(k, x)
            .vectorize(x, vec);

        unnormalized_kernel_y
            // TODO: for debugging
            .compute_root()
            // .compute_at(kernel_y, y)
            .vectorize(y, vec);
        kernel_sum_y
            .compute_at(kernel_y, y)
            .vectorize(y);
        kernel_y
            // TODO: for debugging
            .compute_at(output, y)
            // .compute_root()
            .reorder(k, y)
            .vectorize(y, vec);

        if (upsample) {
            // TODO
            output
                .tile(x, y, xi, yi, 16, 64)
                .parallel(y)
                .vectorize(xi);
            resized_x
                .compute_at(output, x)
                .hoist_storage(output, y)
                .vectorize(x);
            resized_y
                .compute_at(output, xi)
                .unroll(c);
        } else {
            output
                .specialize(input.dim(0).extent() % block_size == 0)
                .tile(x, y, xi, yi, 16, 16)
                .fuse(xi, yi, z)
                .split(z, zo, zi, 32)
                // zi correspond to two 16-element contiguous vectors
                .reorder(zo, zi, x, y, c)
                .gpu_threads(zi)
                .gpu_blocks(c, y);
            output.specialize_fail("we assume image width is divisible by 16");

            as_float.compute_at(output, y);
            kernel_blocks_y.compute_at(resized_y, ry.y)
                .fuse(xi, yi, z)
                .split(z, zo, zi, 32)
                // zi correspond to two 16-element contiguous vectors
                .reorder(zo, zi)
                .gpu_threads(zi);

            resized_y
                .in()
                .compute_at(output, y)
                .vectorize(xi, 16)
                .vectorize(yi, 16);
            resized_y
                .compute_at(resized_y.in(), xo)
                .store_in(MemoryType::WMMAAccumulator);
            resized_y
                .vectorize(xi, 16)
                .vectorize(yi, 16);
            resized_y
                .update()
                .reorder(ry.x, xi, yi, ry.y, xo, yo, c)
                .atomic()
                .vectorize(ry.x)
                .vectorize(xi)
                .vectorize(yi);

            resized_yf16
                .compute_at(output, y);

            resized_x
                .in()
                .compute_at(output, x)
                .reorder(xi, yi, xo, yo, c)
                .vectorize(yi)
                .vectorize(xi);
            
            resized_x
                .compute_at(resized_x.in(), xo)
                .store_in(MemoryType::WMMAAccumulator);
            resized_x
                .vectorize(yi, 16)
                .vectorize(xi, 16);
            resized_x
                .update()
                .reorder(rx.x, xi, yi, rx.y, xo, yo, c)
                .atomic()
                .vectorize(rx.x)
                .vectorize(xi)
                .vectorize(yi);

            kernel_blocks_x.compute_at(resized_x, rx.y)
                .fuse(xi, yi, z)
                .split(z, zo, zi, 32)
                // zi correspond to two 16-element contiguous vectors
                .reorder(zo, zi)
                .gpu_threads(zi);
            output.print_loop_nest();
            // exit(1);
        }

        output.dim(0).set_stride(1);
        output.dim(0).set_min(0);
        input.dim(0).set_stride(1);
    }
};

HALIDE_REGISTER_GENERATOR(Resize, resize);
