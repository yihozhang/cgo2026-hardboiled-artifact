#include "Halide.h"

namespace {

class HalideBlur : public Halide::Generator<HalideBlur> {
public:
    GeneratorParam<int> tile_x{"tile_x", 32};  // X tile.
    GeneratorParam<int> tile_y{"tile_y", 8};   // Y tile.

    Input<Buffer<uint16_t, 2>> input{"input"};
    Output<Buffer<uint16_t, 2>> blur_y{"blur_y"};

    Var x{"x"}, y{"y"}, xi{"xi"}, yi{"yi"};
    Func blur_x{"blur_x"};

    void generate() {
        // To write performant programs in Halide, there are two parts: algorithms and schedules.
        // Halide algorithms are pure pipeline definitions over grids ("tensors")
        // - Execution order and storage are not unspecified
        // - No explicit loops or arrays
        // - Not Turing-complete
        // 
        // Box blur has two stages, but real applications can have hundreds of stages.
        blur_x(x, y) = (input(x, y) + input(x + 1, y) + input(x + 2, y)) / 3;
        blur_y(x, y) = (blur_x(x, y) + blur_x(x, y + 1) + blur_x(x, y + 2)) / 3;

        // A schedule defines 
        //   (1) how should a stage be computed (intra-stage)
        //   (2) when should a stage be computed (inter-stage) 

        // By default, Halide aggressively inlines the stages. 
        // So the default schedule correspond to 
        //
        //   for y
        //     for x
        //       blur_y[x, y] = ... 3x3 grid of input
        // 
        
        // This is inefficient because there's a lot of redundant computation.
        // We can require blur_x to be materialized.
        // 
        //    blur_x.compute_root();
        
        
        
        blur_y.print_loop_nest();
    }



    void generate_gpu() {
        int factor = sizeof(int) / sizeof(short);
        Var y_inner("y_inner");
        blur_y.vectorize(x, factor)
            .split(y, y, y_inner, tile_y)
            .reorder(y_inner, x)
            .unroll(y_inner)
            .gpu_tile(x, y, xi, yi, tile_x, 1);
    }

    void generate_hexagon() {
        const int vector_size = 128;

        blur_y.compute_root()
            .hexagon()
            .prefetch(input, y, y, 2)
            .split(y, y, yi, 128)
            .parallel(y)
            .vectorize(x, vector_size * 2);
        blur_x
            .store_at(blur_y, y)
            .compute_at(blur_y, yi)
            .vectorize(x, vector_size);
    }
};

}  // namespace

HALIDE_REGISTER_GENERATOR(HalideBlur, halide_blur)
