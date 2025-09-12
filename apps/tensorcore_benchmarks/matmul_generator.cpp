#include "Halide.h"
#include <stdio.h>

#include "common.h"

using namespace Halide;

class MatMul : public Halide::Generator<MatMul> {
public:
    // Generator Params
    GeneratorParam<Schedule> gpu_schedule{"gpu_schedule", Schedule::CUDA, {{"cudaonly", Schedule::CUDA}, {"tensorcore", Schedule::TensorCore}}};

    GeneratorParam<int> M{"M", 1024};
    GeneratorParam<int> N{"N", 1024};
    GeneratorParam<int> K{"K", 1024};

    // Inputs
    Input<Buffer<float16_t>> matA{"matA", 2};  // K x M
    Input<Buffer<float16_t>> matB{"matB", 2};  // N x K

    // Output
    Output<Buffer<float>> output{"output", 2};

    void generate() {
        k = RDom(0, K, "k");

        prod(x, y) = cast<float>(0);
        prod(x, y) += cast<float>(matA(k, y)) * cast<float>(matB(x, k));

        output(x, y) = prod(x, y);
    }

    void schedule() {
        // Set dimensions and alignment
        matA.dim(1).set_bounds(0, M);
        matA.dim(1).set_stride(K);

        matA.dim(0).set_bounds(0, K);
        matA.dim(0).set_stride(1);

        matB.dim(1).set_bounds(0, K);
        matB.dim(1).set_stride(N);

        matB.dim(0).set_bounds(0, N);
        matB.dim(0).set_stride(1);

        if (gpu_schedule == Schedule::CUDA) {
            // Schedule taken from cuda_mat_mul app
            Var xi, yi, xio, xii, yii, xo, yo, x_pair, xiio, ty;
            RVar rxo, rxi;

            output.bound(x, 0, N)
                .bound(y, 0, M)
                .tile(x, y, xi, yi, 64, 16)
                .tile(xi, yi, xii, yii, 4, 8)
                .gpu_blocks(x, y)
                .gpu_threads(xi, yi)
                .unroll(xii)
                .unroll(yii);

            prod.compute_at(output, xi)
                .vectorize(x)
                .unroll(y)
                .update()
                .reorder(x, y, k)
                .vectorize(x)
                .unroll(y)
                .unroll(k, 8);

            matA.in().compute_at(prod, k).vectorize(_0).unroll(_1);
            matB.in().compute_at(prod, k).vectorize(_0).unroll(_1);
        } else if (gpu_schedule == Schedule::TensorCore) {
            int tile_x = 16;
            int tile_y = 16;
            int tile_r = 16;

            Var xi, yi, mmxi, mmyi, rxi, ryi;
            RVar rro, rri, rroo;

            output.split(x, x, xi, tile_x * 4)
                .split(xi, xi, mmxi, tile_x)
                .split(y, y, yi, tile_y * 4)
                .split(yi, yi, mmyi, tile_y)
                .gpu_blocks(x, y)
                .reorder({mmxi, mmyi, xi, yi, x, y})
                .unroll(xi)
                .unroll(yi)
                .vectorize(mmxi)
                .vectorize(mmyi);

            // initialization
            prod.compute_at(output, x)
                .store_in(MemoryType::WMMAAccumulator)
                .split(x, x, rxi, tile_x)
                .split(y, y, ryi, tile_y)
                .vectorize(rxi)
                .vectorize(ryi)
                .unroll(x)
                .unroll(y);

            prod.update()
                .split(x, x, rxi, tile_x)
                .split(y, y, ryi, tile_y)
                .split(k, rro, rri, tile_r)
                .split(rro, rro, rroo, 4)
                .reorder({rri, x, y, rroo, rro})
                .unroll(rroo)
                .unroll(x)
                .unroll(y)
                .atomic()
                .vectorize(rri)
                .vectorize(rxi)
                .vectorize(ryi);
        }
    }

private:
    Var x{"x"}, y{"y"};
    RDom k;
    Func prod{"prod"};
};

HALIDE_REGISTER_GENERATOR(MatMul, matmul)
