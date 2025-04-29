#include "Halide.h"
#include "halide_benchmark.h"
#include "halide_test_dirs.h"
#include "matrix_generator.h"

#include <iomanip>
#include <iostream>

using namespace Halide;

bool conv1d(Halide::Target target) {
    (void)target;

    const int acc = 128;

    Var x("x"), y("y");
    ImageParam A(Float(16), 1, "lhs");
    ImageParam B(Float(16), 2, "rhs");

    RDom r(0, acc, "acc");

    Func conv("conv");

    bool use_gpu = false;

    use_gpu = true;

    Var xi("xi"), yi("yi");
    Var rxi("rxi"), ryi("ryi");
    RVar rri("rri"), rro("rro"), rroo("rroo");
    Var mmxi("mmxi"),
        mmyi("mmyi");
    RVar mmri("mmri");
    Var xy("xy"), xyi("xyi");

    conv(x, y) = cast<float>(0);
    conv(x, y) += cast<float>(A(r.x)) * cast<float>(B(x + r.x, y));
    // conv(x, y) += A(u-x) * B(u, y)
    // conv(x, y) += A'(x, u) * B(u, y)
    // 8 x 32 x 16


    int schedule = 2;
    if (schedule == 0) {
        // naive WMMA schedule
        int tile_x = 8;
        int tile_y = 32;
        int tile_r = 8;

        // update
        conv.compute_at(conv.in(), x)
            .store_in(MemoryType::WMMAAccumulator)
            .update()
            .split(x, x, rxi, tile_x)
            .split(y, y, ryi, tile_y)
            .split(r.x, rro, rri, tile_r)
            // I cannot unroll rro, since the temporary buffer refers to
            // rro but is lifted to the host, where rro is not available.
            // .unroll(rro)
            .unroll(y)
            .reorder({rri, rxi, ryi, x, y, rro})
            .atomic()
            .vectorize(rri)
            .vectorize(rxi)
            .vectorize(ryi);

        // initialization
        conv.split(x, x, rxi, tile_x)
            .split(y, y, ryi, tile_y)
            .vectorize(rxi)
            .vectorize(ryi)
            .unroll(y);

        conv.in()
            .split(x, x, xi, tile_x * 1)
            .split(xi, xi, mmxi, tile_x)
            .split(y, y, yi, tile_y * 2)
            .split(yi, yi, mmyi, tile_y)
            .gpu_blocks(x, y)
            .reorder({mmxi, mmyi, xi, yi, x, y})
            .unroll(xi)
            .unroll(yi)
            .vectorize(mmxi)
            .vectorize(mmyi);
    } else if (schedule == 1) {
        int tile_x = 8;
        int tile_y = 32;
        int tile_r = 8;

        // update
        conv.compute_at(conv.in(), x)
            .store_in(MemoryType::WMMAAccumulator)
            .update()
            .split(x, x, rxi, tile_x)
            .split(y, y, ryi, tile_y)
            .split(r.x, rro, rri, tile_r)
            // I cannot unroll rro, since the temporary buffer refers to
            // rro but is lifted to the host, where rro is not available.
            // .unroll(rro)
            .unroll(y)
            .reorder({rri, rxi, ryi, x, y, rro})
            .atomic()
            .vectorize(rri)
            .vectorize(rxi)
            .vectorize(ryi);
        B.in().compute_at(conv, x).store_in(MemoryType::WMMAA).vectorize(_0).vectorize(_1);
        // initialization
        conv.split(x, x, rxi, tile_x)
            .split(y, y, ryi, tile_y)
            .vectorize(rxi)
            .vectorize(ryi)
            .unroll(y);

        conv.in()
            .split(x, x, xi, tile_x * 1)
            .split(xi, xi, mmxi, tile_x)
            .split(y, y, yi, tile_y * 2)
            .split(yi, yi, mmyi, tile_y)
            .gpu_blocks(x, y)
            .reorder({mmxi, mmyi, xi, yi, x, y})
            .unroll(xi)
            .unroll(yi)
            .vectorize(mmxi)
            .vectorize(mmyi);
    }

    Func result = conv.in();

    // result.compile_to_lowered_stmt("/tmp/matmul_flat_1x1.html", {A, B}, HTML, target);

    int row = 4096;
    int col = 4096;
    Buffer<float16_t> b_buf(col, row);
    fill_buffer_flat(b_buf, row / 2, col / 2);
    B.set(b_buf);

    Buffer<float16_t> a_buf(acc);
    for (int i = 0; i < acc; i++) {
        a_buf(i) = float16_t(i);
    }
    A.set(a_buf);

    // NB: if col is 7 (whcih it is supposed to be), then the CUDA kernel
    // crashes with "misaligned address"
    // This is another question to ask during the meeting that why the "residual"
    // part is not computed outside of TensorCore.
    // Buffer<float> out(col - acc, row);
    Buffer<float> out(col - acc, row);
    // out.crop(0, 0, col - acc + 1);
    auto time = Tools::benchmark(5, 5, [&]() {
        result.realize(out, target);
        if (use_gpu) {
            out.device_sync();
        }
    });

    if (use_gpu) {
        out.copy_to_host();
    }

    if (1) {
        for (int j = 0; j < row; ++j) {
            // for (int j = 0; j < 64; ++j) {
            for (int i = 0; i < col - acc; ++i) {
                // for (int i = 0; i < 64; ++i) {
                // std::cerr << out(i, j) << " ";
                float val = 0;
                for (int k = 0; k < acc; ++k) {
                    val += float(a_buf(k)) * float(b_buf(i + k, j));
                }
                if (fabs(val - out(i, j)) > 0.001) {
                    std::cerr << "Invalid result at " << i << ", " << j << "\n"
                              << out(i, j) << " != " << val << "\n";
                    return false;
                }
            }
            // std::cerr << "\n";
        }
    }

    std::cout << "Exec time: " << time << "\n";
    std::cout << "Success!\n";
    return true;
}

bool conv2d(Halide::Target target) {
    (void)target;

    const int acc = 32;

    Var x("x"), y("y");
    ImageParam A(Float(16), 2, "lhs");
    ImageParam B(Float(16), 2, "rhs");

    RDom r(0, acc, 0, acc, "acc");

    Func conv("conv");

    bool use_gpu = false;

    use_gpu = true;

    Var xi("xi"), yi("yi");
    RVar rxi("rxi"), ryi("ryi");
    Var mmxi("mmxi"),
        mmyi("mmyi");
    RVar mmri("mmri");
    Var xy("xy"), xyi("xyi");


    conv(x, y) = cast<float>(0);
    conv(x, y) += cast<float>(A(r.x, r.y)) * cast<float>(B(x + r.x, y + r.y));

    int schedule = 1;
    if (schedule == 0) {
        int tile_x = 8;
        int tile_y = 32;
        int tile_rx = 8;
        int tile_ry = 1;

        // Can't do this because this is scheduling A, not the matrix for A
        // A.in().compute_at(conv, x).store_in(MemoryType::WMMAA);
        // update
        conv.compute_at(conv.in(), x)
            .store_in(MemoryType::WMMAAccumulator)
            .update()
            .tile(x, y, mmxi, mmyi, tile_x, tile_y)
            .tile(r.x, r.y, rxi, ryi, tile_rx, tile_ry)
            .reorder({rxi, mmxi, mmyi, ryi, r.x, r.y, x, y})
            .unroll(ryi)
            .atomic()
            .vectorize(rxi)
            .vectorize(mmxi)
            .vectorize(mmyi);

        // initialization
        conv.split(x, x, mmxi, tile_x)
            .split(y, y, mmyi, tile_y)
            .vectorize(mmxi)
            .vectorize(mmyi)
            .unroll(y);

        conv.in()
            .split(x, x, xi, tile_x)
            .split(xi, xi, mmxi, tile_x)
            .split(y, y, yi, tile_y)
            .split(yi, yi, mmyi, tile_y)
            .gpu_blocks(x, y)
            .reorder({mmxi, mmyi, xi, yi, x, y})
            .unroll(xi)
            .unroll(yi)
            .vectorize(mmxi)
            .vectorize(mmyi);
    } else if (schedule == 1) {
        int tile_x = 4;
        int tile_y = 4;
        int tile_rx = 4;
        int tile_ry = 4;

        // update
        conv.compute_at(conv.in(), xi)
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
        conv.in()
            .tile(x, y, mmxi, mmyi, tile_x, tile_y)
            .gpu_tile(x, y, xi, yi, 16, 16)
            .reorder({mmxi, mmyi, xi, yi, x, y})
            .gpu_blocks(x, y)
            .gpu_threads(xi, yi)
            .unroll(mmxi)
            .unroll(mmyi);
    }

    Func result = conv.in();

    // result.compile_to_lowered_stmt("/tmp/matmul_flat_1x1.html", {A, B}, HTML, target);

    int row = 4096;
    int col = 4096;
    Buffer<float16_t> b_buf(col, row);
    fill_buffer_flat(b_buf, row / 2, col / 2);
    B.set(b_buf);

    Buffer<float16_t> a_buf(acc, acc);
    for (int i = 0; i < acc; i++) {
        for (int j = 0; j < acc; j++) {
            a_buf(i, j) = float16_t(i + j);
        }
    }
    A.set(a_buf);

    Buffer<float> out(col - acc, row - acc);
    auto time = Tools::benchmark(5, 5, [&]() {
        result.realize(out, target);
        if (use_gpu) {
            out.device_sync();
        }
    });

    if (use_gpu) {
        out.copy_to_host();
    }

    if (1) {
        // for (int j = 0; j < row; ++j) {
        for (int j = 0; j < 64; ++j) {
            // for (int i = 0; i < col - acc; ++i) {
            for (int i = 0; i < 64; ++i) {
                // std::cerr << out(i, j) << " ";
                float val = 0;
                for (int k1 = 0; k1 < acc; ++k1) {
                    for (int k2 = 0; k2 < acc; ++k2) {
                        val += float(a_buf(k1, k2)) * float(b_buf(i + k1, j + k2));
                    }
                }
                if (fabs(val - out(i, j)) > 0.001) {
                    std::cerr << "Invalid result at " << i << ", " << j << "\n"
                              << out(i, j) << " != " << val << "\n";
                    return false;
                }
            }
            // std::cerr << "\n";
        }
    }

    std::cout << "Exec time: " << time << "\n";
    std::cout << "Success!\n";
    return true;
}

int main(int argc, char **argv) {
    freopen("/tmp/matmul_flat_1x1.log", "w", stderr);
    Target target = get_target_from_environment().with_feature(Target::CUDA).with_feature(Target::CUDACapability75)
        // .with_feature(Target::Debug)
        ;
    // Target target = get_jit_target_from_environment();
    std::cout << target;

    // printf("Running convolution 1d \n");
    // conv1d(target);
    printf("Running convolution 2d \n");
    conv2d(target);
    return 0;
}
