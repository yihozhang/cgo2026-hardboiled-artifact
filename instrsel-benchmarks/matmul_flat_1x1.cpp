#include "Halide.h"
#include "halide_benchmark.h"
#include "halide_test_dirs.h"

#include <iomanip>
#include <iostream>

using namespace Halide;

template<typename T>
void fill_buffer_flat(Buffer<T> &buf, int row, int acc) {
    for (int iy = 0; iy < row; ++iy) {
        for (int ix = 0; ix < acc; ++ix) {
            T val = T(rand() % 2);
            buf(ix, iy) = val;
        }
    }
}

bool matmul_bf16(Halide::Target target) {
    (void)target;

    const int acc = 128;

    Var x("x"), y("y");
    ImageParam A_input(Float(32), 2, "lhs");
    ImageParam B_input(Float(32), 2, "rhs");

    RDom r(0, acc, "acc");

    Func mm("matmul");

    Func A("A");
    Func B("B");
    A(x, y) = cast<bfloat16_t>(A_input(x, y));
    B(x, y) = cast<bfloat16_t>(B_input(x, y));

    mm(x, y) = cast<float>(0);
    mm(x, y) += cast<float>(cast<float>(A(r.x, y))) * cast<float>(B(x, r.x));
    int tile_x = 16;
    int tile_y = 16;
    int tile_r = 32;
    Var rxi("rxi"), ryi("ryi");
    RVar rri("rri"), rro("rro");

    A.compute_root().tile(x, y, rxi, ryi, tile_x, tile_y);
    B.compute_root().tile(x, y, rxi, ryi, tile_x, tile_y);

    mm.compute_at(mm.in(), x)
        .store_in(MemoryType::AMXTile)
        .update()
        .tile(x, y, rxi, ryi, tile_x, tile_y, TailStrategy::GuardWithIf)
        .split(r.x, rro, rri, tile_r)
        .reorder({rri, rxi, ryi, rro, x, y})
        .atomic()
        .vectorize(rri)
        .vectorize(rxi)
        .vectorize(ryi);

    Var ixi("ixi"), iyi("iyi");
    mm.compute_at(mm.in(), x)
        .tile(x, y, ixi, iyi, tile_x, tile_y)
        .vectorize(ixi)
        .vectorize(iyi);

    // schedule the consumer
    Var mmxi("mmxi"), mmyi("mmyi");
    mm.in()
        .tile(x, y, mmxi, mmyi, tile_x, tile_y)
        .vectorize(mmxi)
        .vectorize(mmyi);

    Func result = mm.in();

    // result.compile_to_lowered_stmt("/tmp/matmul_flat_1x1.html", {A_input, B_input}, HTML, target);

    // test
    int row = 64;
    int col = 64;
    Buffer<float> a_buf(acc, row);
    fill_buffer_flat(a_buf, row, acc);
    A_input.set(a_buf);

    Buffer<float> b_buf(col, acc);
    fill_buffer_flat(b_buf, acc, col);
    B_input.set(b_buf);

    Buffer<float> out(col, row);
    result.realize(out, target);

    std::cout << "Success!\n";
    return true;
}

int main(int argc, char **argv) {
    freopen("/tmp/matmul_flat_1x1.log", "w", stderr);
    Target target("x86-64-linux-avx512_sapphirerapids");
    std::cout << target;

    printf("Running AMX (bf16)\n");
    matmul_bf16(target);
    return 0;
}
