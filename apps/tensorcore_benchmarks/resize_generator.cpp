#include "Halide.h"
#include <stdio.h>

#include "Halide.h"
#include "common.h"

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
    Expr value = sinc(x) * sinc(x / 5);
    value = select(x == 0.0f, 1.0f, value);        // Take care of singularity at zero
    value = select(x > 5 || x < -5, 0.0f, value);  // Clamp to zero out of bounds
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
    {"lanczos", 10, kernel_lanczos}};

class Resize : public Halide::Generator<Resize> {
public:
    GeneratorParam<InterpolationType> interpolation_type{"interpolation_type", Lanczos, {{"box", Box}, {"linear", Linear}, {"cubic", Cubic}, {"lanczos", Lanczos}}};

    // If we statically know whether we're upsampling or downsampling,
    // we can generate different pipelines (we want to reorder the
    // resample in x and in y).
    GeneratorParam<bool> upsample{"upsample", false};
    GeneratorParam<Schedule> gpu_schedule{
        "gpu_schedule", Schedule::CUDA,         //
        {                                       //
         {"cuda_only", Schedule::CUDA},         //
         {"tensorcore", Schedule::TensorCore}}  //
    };

    Input<Buffer<float16_t, 3>> input{"input"};
    Input<float> scale_factor{"scale_factor"};
    Output<Buffer<float16_t, 3>> output{"output"};

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
    bool strict = false;  // true;

    // For a given output x coord, what is the first x coord in the input that
    // we depend on?
    Expr begin_of(Expr x) {
        Expr sourcex = (x + 0.5f) * inverse_scale_factor - 0.5f;

        Expr beginx;
        if (strict) {
            beginx = cast<int>(strict_float(ceil(sourcex - kernel_radius)));
        } else {
            beginx = cast<int>(ceil(sourcex - kernel_radius));
        }

        return beginx;
    }

    void generate() {

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

        // The relationship between an input row and an output row is linear, so
        // it can be represented as a matrix. The matrix is very large and
        // mostly zero however, so we never want to explicitly materialize
        // it. Each row of the matrix contains a small contiguous number of
        // non-zeros. We'll therefore represent only a contiguous region, and
        // remember the column it starts at as a separate expression. We can
        // start at any column we like! We just need to ensure we store enough
        // columns. We'll make the start the same for each group of 16 rows plus
        // linear function of the residual, to make tiling easier (the
        // source address becomes integer-linear in the inner loop).
        constexpr int tile = 16;

        // TODO: Setting this to zero makes it a matmul, but setting it as below
        // makes the resized_y kernel 2x faster for a high-quality resizing kernel
        Expr integer_stride = 0;  // cast<int>(floor(inverse_scale_factor));
        Expr beginx = begin_of((x / tile) * tile) + (x % tile) * integer_stride;
        Expr beginy = begin_of((y / tile) * tile) + (y % tile) * integer_stride;

        // Moving beginx back like this means we need to represent a longer
        // contiguous region.
        Expr extra_zeros = begin_of(tile) - begin_of(0) - tile * integer_stride;

        // We'll also round up the output to the next multiple of 16.
        Expr span = ((kernel_taps + extra_zeros + tile - 1) / tile) * tile;

        // Don't go off the end of the image. Those columns would be zero
        // anyway.
        beginx = clamp(beginx, 0, input.width() - span);
        beginy = clamp(beginy, 0, input.height() - span);

        r = {0, span, "r"};
        const KernelInfo &info = kernel_info[interpolation_type];

        unnormalized_kernel_x(x, k) = info.kernel((k + beginx - sourcex) * kernel_scaling);
        unnormalized_kernel_y(y, k) = info.kernel((k + beginy - sourcey) * kernel_scaling);

        kernel_sum_x(x) += unnormalized_kernel_x(x, r);
        kernel_sum_y(y) += unnormalized_kernel_y(y, r);

        kernel_x(x, k) = cast<float16_t>(unnormalized_kernel_x(x, k) / kernel_sum_x(x));
        kernel_y(y, k) = cast<float16_t>(unnormalized_kernel_y(y, k) / kernel_sum_y(y));

        resized_y(x, y, c) = 0.f;
        resized_y(x, y, c) += cast<float>(kernel_y(y, r)) * cast<float>(as_float(x, r + beginy, c));

        Func resized_y_f16{"resized_y_f16"};
        resized_y_f16(x, y, c) = cast<float16_t>(resized_y(x, y, c));

        resized_x(x, y, c) = 0.f;
        resized_x(x, y, c) += cast<float>(kernel_x(x, r)) * cast<float>(resized_y_f16(r + beginx, y, c));

        output(x, y, c) = cast<float16_t>(clamp(resized_x(x, y, c), 0.f, 1.f));

        // Schedule

        Var xi("xi"), yi("yi"), ki("ki");

        // Precompute the sparse matrices
        kernel_x
            .compute_root()
            .gpu_tile(x, k, xi, ki, 32, 8);
        unnormalized_kernel_x
            .compute_root()
            .gpu_tile(x, k, xi, ki, 32, 8);
        kernel_sum_x.in()
            .compute_root()
            .gpu_tile(x, xi, 32);

        kernel_y
            .compute_root()
            .gpu_tile(y, k, yi, ki, 32, 8);
        unnormalized_kernel_y
            .compute_root()
            .gpu_tile(y, k, yi, ki, 32, 8);
        kernel_sum_y.in()
            .compute_root()
            .gpu_tile(y, yi, 32);

        // For large downsamples, this is the expensive stage. When resizing in
        // y, the load from the kernel doesn't depend on x and c, and the load from
        // the image doesn't depend on y % 16, so we can schedule it like a
        // matrix multiply (i.e. tile it).
        Var xii{"xii"}, yii{"yii"};
        resized_y_f16
            .compute_root()
            .align_bounds(x, 16)
            .align_bounds(y, 16)
            .reorder(c, x, y)
            .unroll(c)
            .gpu_tile(x, y, xi, yi, 64, 32, TailStrategy::RoundUp)
            .tile(xi, yi, xii, yii, 4, 4)
            .unroll(xii)
            .unroll(yii);
        resized_y
            .compute_at(resized_y_f16, xi)
            .unroll(c)
            .unroll(x)
            .unroll(y)
            .update()
            .reorder(x, y, c, r)
            .unroll(c)
            .unroll(x)
            .unroll(y);
        as_float.compute_at(resized_y, c).vectorize(x).vectorize(y);
        kernel_y.in().compute_at(resized_y, r).vectorize(y).vectorize(k);

        // For large downsamples, it's hard to fill the machine, because we've
        // already downsampled in y. We'll use much smaller tiles.
        output
            .compute_root()
            .align_bounds(x, 16)
            .align_bounds(y, 16)
            .reorder(c, x, y)
            .unroll(c)
            .never_partition(x, y)
            .gpu_tile(x, y, xi, yi, 32, 4, TailStrategy::RoundUp)
            .tile(xi, yi, xii, yii, 2, 2)
            .vectorize(xii)
            .unroll(yii);

        resized_x
            .compute_at(output, xi)
            .unroll(c)
            .unroll(x)
            .unroll(y)
            .update()
            .reorder(x, y, c, r)
            .unroll(c)
            .unroll(x)
            .unroll(y);
        resized_y_f16.in().compute_at(resized_x, c).vectorize(x).vectorize(y);
        kernel_x.in().compute_at(resized_x, r).vectorize(x).vectorize(k);

        output.dim(0).set_min(0);
        output.dim(1).set_min(0);
        output.dim(2).set_bounds(0, 3);
        input.dim(0).set_min(0);
        input.dim(1).set_min(0);
        input.dim(2).set_bounds(0, 3);
    }
};

HALIDE_REGISTER_GENERATOR(Resize, resize);
