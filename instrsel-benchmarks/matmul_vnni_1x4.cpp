#include "Halide.h"
#include "halide_benchmark.h"
#include "halide_test_dirs.h"

#include <iomanip>
#include <iostream>

using namespace Halide;

void fill_buffer_a(Buffer<bfloat16_t> &buf, int row, int acc) {
    for (int iy = 0; iy < row; ++iy) {
        for (int ix = 0; ix < acc; ++ix) {
            // value between 0 and 100
            bfloat16_t val = bfloat16_t(((float)rand() / (float)(RAND_MAX)) * 100.f);
            buf(ix, iy) = val;
        }
    }
}

void fill_buffer_b(Buffer<bfloat16_t> &buf, int col, int acc) {
    for (int iy = 0; iy < acc / 2; ++iy) {
        for (int ix = 0; ix < col; ++ix) {
            for (int ik = 0; ik < 2; ++ik) {
                bfloat16_t val = bfloat16_t(((float)rand() / (float)(RAND_MAX)) * 100.f);
                buf(ik, ix, iy) = val;
            }
        }
    }
}

bool matmul_bf16(Halide::Target target) {
    (void)target;

    // lhs: 32x16, rhs: 16x32
    const int acc = 4096;
    const int X_ACC = 1;
    const int Y_ACC = 2;

    Var x("x"), y("y");
    ImageParam A(BFloat(16), 2, "lhs");
    ImageParam B(BFloat(16), 3, "rhs");

    B.dim(1).set_stride(2);

    RDom r(0, acc, "acc");

    Func mm("matmul");
    mm(x, y) = cast<float>(0);
    mm(x, y) += cast<float>(cast<float>(A(r.x, y))) * cast<float>(B(r.x % 2, x, r.x / 2));

    int tile_x = 8;
    int tile_y = 8;
    int tile_r = 4;


    Var cx("cx"), cy("cy");
    Var rxi("rxi"), ryi("ryi");
    // Var rvxi("rvxi"), rvyi("rvyi");
    RVar rri("rri"), rro("rro");

    mm.compute_at(mm.in(), x)
        .store_in(MemoryType::AMXTile)
        .update()
        .tile(x, y, cx, cy, X_ACC * tile_x, Y_ACC * tile_y, TailStrategy::GuardWithIf)
        .tile(cx, cy, rxi, ryi, tile_x, tile_y)
        .split(r.x, rro, rri, tile_r)
        .reorder({rri, rxi, ryi, cy, cx, rro, x, y})
        .unroll(cx)
        .unroll(cy)
        .atomic()
        .vectorize(rri)
        .vectorize(rxi)
        .vectorize(ryi);

    Var ixi("ixi"), iyi("iyi");
    mm.compute_at(mm.in(), x)
        .tile(x, y, cx, cy, X_ACC * tile_x, Y_ACC * tile_y)
        .tile(cx, cy, ixi, iyi, tile_x, tile_y)
        // .tile(ixi, iyi, rvxi, rvyi, tile_x, tile_y)
        .unroll(cx)
        .unroll(cy)
        .vectorize(ixi)
        .vectorize(iyi);

    // schedule the consumer
    Var mmxi("mmxi"), mmyi("mmyi");
    mm.in()
        .tile(x, y, cx, cy, X_ACC * tile_x, Y_ACC * tile_y)
        .tile(cx, cy, mmxi, mmyi, tile_x, tile_y)
        .reorder(mmxi, mmyi, cy, cx, x, y)
        .unroll(cx)
        .unroll(cy)
        .vectorize(mmxi)
        .vectorize(mmyi);

    // mm.in().output_buffer().dim(2).set_bounds(0, 2);
    // mm.in().output_buffer().dim(3).set_bounds(0, 2);

    Func result = mm.in();

    int row = 32, col = 32;
    Buffer<bfloat16_t> A_buf(acc, row);
    Buffer<bfloat16_t> B_buf(2, col, acc / 2);
    fill_buffer_a(A_buf, row, acc);
    fill_buffer_b(B_buf, col, acc);
    A.set(A_buf);
    B.set(B_buf);
    Buffer<float> out(col, row);
    result.realize(out, target);

    // Uncomment to check the asm
    // result.compile_to_llvm_assembly(Internal::get_test_tmp_dir() + "tiled_matmul_bf16.ll", {A, B}, target);
    // result.compile_to_assembly(Internal::get_test_tmp_dir() + "tiled_matmul.s", {A, B}, target);
    result.compile_to_lowered_stmt("/tmp/matmul_preload_vnni.html", {A, B}, HTML, target);

    std::cout << "Success!\n";
    return true;
}

int main(int argc, char **argv) {
    freopen("/tmp/matmul_preload_vnni.log", "w", stderr);
    Target target("x86-64-linux-avx512_sapphirerapids");

    printf("Running AMX (bf16)\n");
    matmul_bf16(target);
    return 0;
}
