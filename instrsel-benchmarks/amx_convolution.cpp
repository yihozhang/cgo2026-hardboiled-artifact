#include "Halide.h"
#include "halide_benchmark.h"
#include "halide_test_dirs.h"
#include "matrix_generator.h"

#include <iomanip>
#include <iostream>

using namespace Halide;

bool conv1d(Halide::Target target, bool check_result) {
    (void)target;

    const int acc = 64;
    const int X_ACC = 1;
    const int Y_ACC = 1;

    Var x("x"), y("y");
    ImageParam A(BFloat(16), 1, "lhs");
    ImageParam B(BFloat(16), 2, "rhs");

    RDom r(0, acc, "acc");

    Var xi("xi"), yi("yi");
    Var rxi("rxi"), ryi("ryi");
    RVar rri("rri"), rro("rro"), rroo("rroo");
    Var mmxi("mmxi"),
        mmyi("mmyi");
    RVar mmri("mmri");
    Var xy("xy"), xyi("xyi");

    Func conv("matmul");
    conv(x, y) = cast<float>(0);
    conv(x, y) += cast<float>(cast<float>(A(r.x))) * cast<float>(B(x + r.x, y));

    int tile_x = 16;
    int tile_y = 16;
    int tile_r = 16;

    conv.compute_at(conv.in(), x)
        .store_in(MemoryType::AMXTile)
        .update()
        .split(x, x, rxi, tile_x)
        .split(y, y, ryi, tile_y)
        .split(r.x, rro, rri, tile_r)
        // .unroll(y)
        .reorder({rri, rxi, ryi, rro, y, x})
        .atomic()
        .vectorize(rxi)
        .vectorize(ryi)
        .vectorize(rri);

    // initialization
    conv.split(x, x, mmxi, tile_x)
        .split(y, y, mmyi, tile_y)
        .vectorize(mmxi)
        .vectorize(mmyi)
        // .unroll(y)
        ;
    conv.in()
        .split(x, x, xi, tile_x * X_ACC)
        .split(xi, xi, mmxi, tile_x)
        .split(y, y, yi, tile_y * Y_ACC)
        .split(yi, yi, mmyi, tile_y)
        .reorder({mmxi, mmyi, xi, yi, x, y})
        // .unroll(xi)
        // .unroll(yi)
        .vectorize(mmxi)
        .vectorize(mmyi);

    Func result = conv.in();

    // result.compile_to_lowered_stmt("/tmp/matmul_flat_1x1.html", {A, B}, HTML, target);

    int row = 512;
    int col = 512;
    Buffer<bfloat16_t> b_buf(col, row);
    fill_buffer_flat(b_buf, row / 2, col / 2);
    B.set(b_buf);

    Buffer<bfloat16_t> a_buf(acc);
    for (int i = 0; i < acc; i++) {
        a_buf(i) = bfloat16_t(i);
    }
    A.set(a_buf);

    Buffer<float> out(col - acc, row);
    auto time = Tools::benchmark(5, 5, [&]() {
        result.realize(out, target);
    });

    if (check_result) {
        // Check the result
        for (int y = 0; y < row; y++) {
            for (int x = 0; x < col - acc; x++) {
                float expected = 0;
                for (int i = 0; i < acc; i++) {
                    expected += float(a_buf(i)) * float(b_buf(x + i, y));
                }
                if (out(x, y) != expected) {
                    std::cerr << "Mismatch at (" << x << ", " << y << "): "
                              << out(x, y) << " != " << expected << std::endl;
                    return false;
                }
            }
        }
    }

    std::cout << "Time: " << time << " ms" << std::endl;
    std::cout << "Result: " << out(0, 0) << std::endl;
    return true;
}

bool conv2d(Halide::Target target, bool check_result) {
    (void)target;

    const int acc = 32;
    const int X_ACC = 1;
    const int Y_ACC = 1;

    Var x("x"), y("y");
    ImageParam A(BFloat(16), 2, "lhs");
    ImageParam B(BFloat(16), 2, "rhs");

    RDom r(0, acc, 0, acc, "acc");

    Var xi("xi"), yi("yi");
    RVar rxi("rxi"), ryi("ryi");
    RVar rri("rri"), rro("rro"), rroo("rroo");
    Var mmxi("mmxi"),
        mmyi("mmyi");
    RVar mmri("mmri");
    Var xy("xy"), xyi("xyi");

    Func conv("matmul");
    conv(x, y) = cast<float>(0);
    conv(x, y) += cast<float>(cast<float>(A(r.x, r.y))) * cast<float>(B(x + r.x, y + r.y));

    int tile_x = 16;
    int tile_y = 16;
    int tile_rx = 16;
    int tile_ry = 2;

    conv.compute_at(conv.in(), x)
        .store_in(MemoryType::AMXTile)
        .update()
        .split(x, x, mmxi, tile_x)
        .split(y, y, mmyi, tile_y)
        .tile(r.x, r.y, rxi, ryi, tile_rx, tile_ry)
        // .unroll(y)
        .reorder({rxi, mmxi, mmyi, ryi, r.x, r.y, y, x})
        .atomic()
        .vectorize(rxi)
        .vectorize(mmxi)
        .vectorize(mmyi);

    // initialization
    conv.split(x, x, mmxi, tile_x)
        .split(y, y, mmyi, tile_y)
        .vectorize(mmxi)
        .vectorize(mmyi)
        // .unroll(y)
        ;
    conv.in()
        .split(x, x, xi, tile_x * X_ACC)
        .split(xi, xi, mmxi, tile_x)
        .split(y, y, yi, tile_y * Y_ACC)
        .split(yi, yi, mmyi, tile_y)
        .reorder({mmxi, mmyi, xi, yi, x, y})
        // .unroll(xi)
        // .unroll(yi)
        .vectorize(mmxi)
        .vectorize(mmyi);

    Func result = conv.in();

    // result.compile_to_lowered_stmt("/tmp/matmul_flat_1x1.html", {A, B}, HTML, target);

    int row = 128;
    int col = 128;
    Buffer<bfloat16_t> b_buf(col, row);
    fill_buffer_flat(b_buf, row / 2, col / 2);
    B.set(b_buf);

    Buffer<bfloat16_t> a_buf(acc, acc);
    for (int i = 0; i < acc; i++) {
        for (int j = 0; j < acc; j++) {
            a_buf(i, j) = bfloat16_t(i + j);
        }
    }
    A.set(a_buf);

    Buffer<float> out(col - acc, row - acc);
    auto time = Tools::benchmark(5, 5, [&]() {
        result.realize(out, target);
    });

    if (check_result) {
        // Check the result
        for (int y = 0; y < row - acc; y++) {
            for (int x = 0; x < col - acc; x++) {
                float expected = 0;
                for (int i = 0; i < acc; i++) {
                    for (int j = 0; j < acc; j++) {
                        expected += float(a_buf(i, j)) * float(b_buf(x + i, y + j));
                    }
                }
                if (out(x, y) != expected) {
                    std::cerr << "Mismatch at (" << x << ", " << y << "): "
                              << out(x, y) << " != " << expected << std::endl;
                    return false;
                }
            }
        }
    }

    std::cout << "Time: " << time << " ms" << std::endl;
    std::cout << "Result: " << out(0, 0) << std::endl;
    return true;
}

int main(int argc, char **argv) {
    freopen("/tmp/amx_convolution.log", "w", stderr);
    Target target("x86-64-linux-avx512_sapphirerapids");

    printf("Running AMX conv1d\n");
    // conv1d(target, true);
    conv2d(target, true);
    return 0;
}
