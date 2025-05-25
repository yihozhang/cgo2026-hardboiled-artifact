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

    Var x("x"), y("y"), z("z");
    ImageParam A(Float(16), 2, "A");
    ImageParam B(Float(16), 2, "B");
    ImageParam C(Float(16), 2, "C");

    RDom r(0, acc, 0, acc, "acc");

    Func mm("matmul");

    if (target.has_feature(halide_target_feature_avx512_sapphirerapids)) {
        mm(x, y) = cast<float>(0);
        // mm(x, w) = A(x, y) * B(y, z) * C(z, w)
        // A(y, z) * B(z, x) * C(x, w)
        mm(x, y) += cast<float>(cast<bfloat16_t>(cast<float>(A(r.x, y)) * cast<float>(B(r.y, r.x)))) * cast<float>(C(x, r.y));

        int tile_x = 4;
        int tile_y = 8;
        int tile_r = 4;
        Var rxi("rxi"), ryi("ryi");
        RVar rxri("rxri"), rxro("rxro");
        RVar ryri("ryri"), ryro("ryro");

        mm.compute_at(mm.in(), x)
            .store_in(MemoryType::AMXTile)
            .update()
            .tile(x, y, rxi, ryi, tile_x, tile_y, TailStrategy::GuardWithIf)
            .split(r.x, rxro, rxri, tile_r)
            .split(r.y, ryro, ryri, tile_r)
            .reorder({rxri, ryri, rxi, ryi, rxro, ryro, x, y})
            .atomic()
            .vectorize(rxri)
            .vectorize(ryri)
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

    } else if (target.get_cuda_capability_lower_bound() >= 75) {

        mm(x, y) = cast<float16_t>(0);
        // mm(x, w) = A(x, y) * B(y, z) * C(z, w)
        // A(y, z) * B(z, x) * C(x, w)
        mm(x, y) += (A(r.x, y) * B(r.y, r.x)) * C(x, r.y);

        int tile_x = 16;
        int tile_y = 16;
        int tile_rx = 16;
        int tile_ry = 16;
        Var xi("xi"), yi("yi");
        Var rxi("rxi"), ryi("ryi");
        RVar rxri("rxri"), rxro("rxro");
        RVar ryri("ryri"), ryro("ryro");

        mm.compute_at(mm.in(), x)
            .store_in(MemoryType::WMMAAccumulator)
            .update()
            .split(x, x, rxi, tile_x)
            .split(y, y, ryi, tile_y)
            .split(r.x, rxro, rxri, tile_rx)
            .split(r.y, ryro, ryri, tile_ry)
            .reorder({rxri, ryri, rxi, ryi, x, y, rxro, ryro})
            // .unroll(rxro)
            // .unroll(ryro)
            // .unroll(x)
            // .unroll(y)
            .atomic()
            .vectorize(rxri)
            .vectorize(ryri)
            .vectorize(rxi)
            .vectorize(ryi);

        Var ixi("ixi"), iyi("iyi");
        mm.split(x, x, rxi, tile_x)
            .split(y, y, ryi, tile_y)
            .vectorize(rxi)
            .vectorize(ryi)
            .unroll(y);

        // schedule the consumer
        Var mmxi("mmxi"), mmyi("mmyi");
        mm.in()
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
    }
    Func result = mm.in();
    // result.compile_to_lowered_stmt("/tmp/matmul_flat_1x1.html", {A, B, C}, HTML, target);

    if (1) {
        bool use_gpu = true;
        int row = 512;
        int col = 512;
        
        Buffer<float16_t> a_buf(acc, row);
        fill_buffer_flat(a_buf, row, acc);
        A.set(a_buf);
        Buffer<float16_t> b_buf(acc, acc);
        fill_buffer_flat(b_buf, acc, acc);
        B.set(b_buf);
        Buffer<float16_t> c_buf(col, acc);
        fill_buffer_flat(c_buf, acc, col);
        C.set(c_buf);



        // NB: if col is 7 (whcih it is supposed to be), then the CUDA kernel
        // crashes with "misaligned address"
        // This is another question to ask during the meeting that why the "residual"
        // part is not computed outside of TensorCore.
        // Buffer<float> out(col - acc, row);
        Buffer<float16_t> out(col, row);
        // out.crop(0, 0, col - acc + 1);
        auto time = Tools::benchmark(5, 5, [&]() {
            result.realize(out, target);
            if (use_gpu) {
                out.device_sync();
            }
        });
        std::cout << "Time: " << time << "ms\n";
    }

    std::cout << "Success!\n";
    return true;
}

int main(int argc, char **argv) {
    freopen("/tmp/matmul_flat_1x1.log", "w", stderr);
    // Target target("x86-64-linux-avx512_sapphirerapids");
    Target target = get_target_from_environment().with_feature(Target::CUDA).with_feature(Target::CUDACapability75)
        // .with_feature(Target::Debug)
        ;

    printf("Running AMX (bf16)\n");
    matmul_bf16(target);
    return 0;
}
