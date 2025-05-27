#include "Halide.h"
#include "halide_benchmark.h"
#include "halide_test_dirs.h"
#include "matrix_generator.h"

#include <iomanip>
#include <iostream>

using namespace Halide;

bool conv1d(Halide::Target target) {
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

    int row = 4096;
    int col = 4096;
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

    std::cout << "Time: " << time << " ms" << std::endl;
    std::cout << "Result: " << out(0, 0) << std::endl;
    return true;
}

int main(int argc, char **argv) {
    freopen("/tmp/amx_convolution.log", "w", stderr);
    Target target("x86-64-linux-avx512_sapphirerapids");

    printf("Running AMX conv1d\n");
    conv1d(target);
    return 0;
}
