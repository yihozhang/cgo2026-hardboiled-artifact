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
        // The algorithm
        blur_x(x, y) = (input(x, y) + input(x + 1, y) + input(x + 2, y)) / 3;
        blur_y(x, y) = (blur_x(x, y) + blur_x(x, y + 1) + blur_x(x, y + 2)) / 3;

        // inline everything 900 MP/s

        // 1200 MP/s
        // blur_x.compute_root();

        // 2600 MP/s
        // blur_x.compute_root();
        // blur_x
        //     .vectorize(x, 16);
        // blur_y
        //     .vectorize(x, 16);

        // 5000 MP/s
        // blur_x.compute_root();
        // blur_x
        //     .split(y, y, yi, 1024)
        //     .parallel(y)
        //     .vectorize(x, 16);
        // blur_y
        //     .split(y, y, yi, 1024)
        //     .parallel(y)
        //     .vectorize(x, 16);

        // 12000 MP/s
        // blur_x.compute_at(blur_y, y);
        // blur_x
        //     .vectorize(x, 16)
        //     .unroll(x, 2);
        // blur_y
        //     .split(y, y, yi, 8)
        //     .parallel(y)
        //     .vectorize(x, 16)
        //     .unroll(x, 2);

        // 14000 MP/s
        // Var yii("yii");
        // blur_x.compute_at(blur_y, x);
        // blur_x
        //     .vectorize(x, 16)
        //     ;
        // blur_y
        //     .tile(x, y, xi, yi, 256, 512)
        //     .split(yi, yi, yii, 16)
        //     .reorder(xi, yii, x, yi, y)
        //     .parallel(y)
        //     .vectorize(xi, 16)
        //     ;
       
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
