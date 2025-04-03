#include "Halide.h"
#include "halide_benchmark.h"
#include "halide_test_dirs.h"
#include "matrix_generator.h"

#include <iomanip>
#include <iostream>

using namespace Halide;

bool matmul_bf16(Halide::Target target) {
    (void)target;

    const int acc = 4096;

    Var x("x"), y("y");
    ImageParam A(Float(16), 1, "lhs");
    ImageParam B(Float(16), 2, "rhs");

    RDom r(0, 8, "acc");

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

    int tile_x = 8;
    int tile_y = 32;

    conv(x, y) = cast<float>(0);
    conv(x, y) += cast<float>(A(r.x)) * cast<float>(B(x + r.x, y));

    // update
    conv.compute_at(conv.in(), x)
        .store_in(MemoryType::WMMAAccumulator)
        .update()
        .split(x, x, rxi, tile_x)
        .split(y, y, ryi, tile_y)
        // .split(r.x, rro, rri, tile_r)
        //
        // .split(rro, rro, rroo, 4)
        .reorder({r.x, rxi, ryi, x, y})
        .atomic()
        .vectorize(r.x)
        .vectorize(rxi)
        .vectorize(ryi);

    // B.in(mm).compute_at(mm, rroo).store_in(MemoryType::WMMAB)
    //     .vectorize(x)
    //     .vectorize(y)
    //     ;

    // initialization
    conv.split(x, x, rxi, tile_x)
        .split(y, y, ryi, tile_y)
        .vectorize(rxi)
        .vectorize(ryi)
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

    Func result = conv.in();

    result.compile_to_lowered_stmt("/tmp/matmul_flat_1x1.html", {A, B}, HTML, target);


    int row = 4096;
    int col = 4096;
    Buffer<float16_t> b_buf(acc, row);
    fill_buffer_flat_one(b_buf, row, acc);
    B.set(b_buf);

    Buffer<float16_t> a_buf(16);
    for (int i = 0; i < 16; i++) {
        a_buf(i) = float16_t(fabs(8 - i));
    }
    A.set(a_buf);

    // NB: if col is 7 (whcih it is supposed to be), then the CUDA kernel
    // crashes with "misaligned address"
    // This is another question to ask during the meeting that why the "residual"
    // part is not computed outside of TensorCore.
    Buffer<float> out(col - 16, row);
    auto time = Tools::benchmark(5, 5, [&]() {
        result.realize(out, target);
        if (use_gpu) {
            out.device_sync();
        }
    });

    std::cout << "Exec time: " << time << "\n";
    std::cout << "Success!\n";
    return true;
}

int main(int argc, char **argv) {
    freopen("/tmp/matmul_flat_1x1.log", "w", stderr);
    // Target target("x86-64-linux-avx512_sapphirerapids");
    // Target target("x86-64-linux-cuda_capability_70");
    Target target = get_target_from_environment().with_feature(Target::CUDA).with_feature(Target::CUDACapability75)
        .with_feature(Target::Debug)
        ;
    // Target target = get_jit_target_from_environment();
    std::cout << target;

    printf("Running AMX (bf16)\n");
    matmul_bf16(target);
    return 0;
}
