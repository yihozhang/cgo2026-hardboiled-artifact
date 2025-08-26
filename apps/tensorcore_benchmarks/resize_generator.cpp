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
    Var xi{"xi"}, yi{"yi"};
    Var xo{"xo"}, yo{"yo"};
    RDom r;
    RVar ri{"ri"}, ro{"ro"};
    RDom rx, ry;

    // Intermediate Funcs
    Func as_float{"as_float"},
        resized_x{"resized_x"},
        resized_y{"resized_y"},
        unnormalized_kernel_x{"unnormalized_kernel_x"},
        unnormalized_kernel_y{"unnormalized_kernel_y"},
        kernel_x{"kernel_x"},
        kernel_y{"kernel_y"},
        kernel_sum_x{"kernel_sum_x"},
        kernel_sum_y{"kernel_sum_y"};
    Func is_empty_block{"is_empty_block"}, kernel_blocks{"kernel_blocks"};

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
        Func resized;
        if (upsample) {
            resized_x(x, y, c) = sum(kernel_x(x, r) * as_float(r + beginx, y, c), "resized_x");
            resized_y(x, y, c) = sum(kernel_y(y, r) * resized_x(x, r + beginy, c), "resized_y");
            resized = resized_y;
        } else {
            // resized_y(x, y, c) = sum(kernel_y(y, r) * as_float(x, r + beginy, c), "resized_y");
            resized_y(x, y, c) = cast<float16_t>(0.f);
            resized_y(x, y, c) += cast<float16_t>(cast<float>(kernel_y(y, r)) * cast<float>(as_float(x, r + beginy, c)));

            // resized_x(x, y, c) = sum(kernel_x(x, r) * resized_y(r + beginx, y, c), "resized_x");
            // resized_x(x, y, c) = 0.f;
            // resized_x(x, y, c) += cast<float>(kernel_x(x, r)) * cast<float>(resized_y(r + beginx, y, c));

            is_empty_block(y, x) = 
                (x * block_size <= begin_of(block_size * y + block_size - 1) + kernel_taps) && 
                begin_of(block_size * y) <= x * block_size + block_size - 1;
            
            // Expr offset_from_beginx = (z + x * block_size) - begin_of(w + y * block_size);
            // z and x denote columns of the band kernel matrix
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
            // kernel_blocks(w, z, y, x) = 
                // select(0 <= offset_from_beginx && offset_from_beginx <= kernel_taps, kernel_x(w + y * block_size, offset_from_beginx), 0);
            Expr offset_from_beginx = y - begin_of(x);
            kernel_blocks(x, y) = 
                select(0 <= offset_from_beginx && offset_from_beginx <= kernel_taps, kernel_x(x, offset_from_beginx), 0);
            
            // rx = {0, input.dim(0).extent(), 0, input.dim(0).extent(), "rx"};
            rx = {0, input.dim(0).extent(), "rx"};
            // rx.where(is_empty_block(x / block_size, rx / block_size));
            resized_x(x, y, c) = 0.f;
            // resized_x(rx.x, y, c) += cast<float>(kernel_blocks(rx.x % block_size, rx.y % block_size, rx.x / block_size, rx.y / block_size)) * cast<float>(resized_y(rx.y, y, c));
            resized_x(x, y, c) += cast<float>(kernel_blocks(x, rx.x)) * cast<float>(resized_y(rx.x, y, c));

            resized = resized_x;
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
            // .compute_at(output, y)
            .compute_root()
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
                .tile(x, y, xi, yi, 16, 16)
                .reorder(xi, yi, x, y, c)
                // .vectorize(xi)
                ;
            resized_y.compute_root();
            RVar rxi("rxi"), rxo("rxo"), ryi("ryi"), ryo("ryo");
            resized_x
                .in()
                .compute_root()
                .tile(x, y, xi, yi, block_size, block_size
                    // TailStrategy::GuardWithIf
                )
                .reorder(xi, yi, x, y, c);
            resized_x
                // .compute_at(output, xi)
                // .compute_root()
                .compute_at(resized_x.in(), x)
                // .store_in(MemoryType::WMMAAccumulator)
                .tile(x, y, xi, yi, 16, 16)
                .reorder(xi, yi, x, y, c)
                .vectorize(yi, 16)
                .vectorize(xi, 16)
                .update()
                // .tile(rx.x, rx.y, rxo, ryo, rxi, ryi, block_size, block_size)
                .split(rx.x, rxo, rxi, block_size)
                .split(y, yo, yi, block_size)
                .split(x, xo, xi, block_size)
                // .split(r, ro, ri, 16)
                .reorder(rxi, xi, yi, rxo, xo, yo, c)
                .atomic()
                // .vectorize(rxi, 16)
                .vectorize(rxi)
                .vectorize(xi)
                .vectorize(yi);
                // .unroll(c)
                ;
            // kernel_blocks.compute_at(resized_x, yi);
            kernel_blocks.compute_root();
        }

        output.dim(0).set_stride(1);
        output.dim(0).set_min(0);
        input.dim(0).set_stride(1);
        // Allow the input and output to have arbitrary memory layout,
        // and add some specializations for a few common cases. If
        // your case is not covered (e.g. planar input, packed rgb
        // output), you could add a new specialization here.
        // output.dim(0).set_stride(Expr());
        // input.dim(0).set_stride(Expr());

        // Expr planar = (output.dim(0).stride() == 1 &&
        //                input.dim(0).stride() == 1);
        // Expr packed_rgb = (output.dim(0).stride() == 3 &&
        //                    output.dim(2).stride() == 1 &&
        //                    output.dim(2).min() == 0 &&
        //                    output.dim(2).extent() == 3 &&
        //                    input.dim(0).stride() == 3 &&
        //                    input.dim(2).stride() == 1 &&
        //                    input.dim(2).min() == 0 &&
        //                    input.dim(2).extent() == 3);
        // Expr packed_rgba = (output.dim(0).stride() == 4 &&
        //                     output.dim(2).stride() == 1 &&
        //                     output.dim(2).min() == 0 &&
        //                     output.dim(2).extent() == 4 &&
        //                     input.dim(0).stride() == 4 &&
        //                     input.dim(2).stride() == 1 &&
        //                     input.dim(2).min() == 0 &&
        //                     input.dim(2).extent() == 4);

        // output.specialize(planar);

        // output.specialize(packed_rgb)
        //     .reorder(c, xi, yi, x, y)
        //     .unroll(c);

        // output.specialize(packed_rgba)
        //     .reorder(c, xi, yi, x, y)
        //     .unroll(c);
    }
};

HALIDE_REGISTER_GENERATOR(Resize, resize);
