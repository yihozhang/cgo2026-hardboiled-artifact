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
    ImageParam A_input(Float(32), 2, "lhs");
    ImageParam B_input(Float(32), 2, "rhs");

    RDom r(0, acc, "acc");

    Func mm("matmul");

    bool use_gpu = false;

    if (target.has_feature(Target::AVX512_SapphireRapids)) {
        Func A("A");
        Func B("B");
        A(x, y) = cast<bfloat16_t>(A_input(x, y));
        B(x, y) = cast<bfloat16_t>(B_input(x, y));

        mm(x, y) = cast<float>(0);
        mm(x, y) += cast<float>(cast<float>(A(r.x, y))) * cast<float>(B(x, r.x));
        int tile_x = 16;
        int tile_y = 32;
        int tile_r = 16;
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
    } else if (target.has_feature(Target::CUDACapability70)) {
        use_gpu = true;

        Func A("A");
        Func B("B");
        Var xi("xi"), yi("yi");
        Var rxi("rxi"), ryi("ryi");
        RVar rri("rri"), rro("rro"), rroo("rroo");
        Var mmxi("mmxi"),
            mmyi("mmyi");
        RVar mmri("mmri");
        Var xy("xy"), xyi("xyi");

        int tile_x = 16;
        int tile_y = 16;
        int tile_r = 16;

        A(x, y) = cast<float16_t>(A_input(x, y));
        B(x, y) = cast<float16_t>(B_input(x, y));

        mm(x, y) = cast<float>(0);
        mm(x, y) += cast<float>(A(r.x, y)) * cast<float>(B(x, r.x));

        int schedule = 0;

        if (schedule == 0) {
            // preload B and unroll
            A.compute_root().gpu_tile(x, y, rxi, ryi, tile_x, tile_y);
            B.compute_root().gpu_tile(x, y, rxi, ryi, tile_x, tile_y);

            // update
            mm.compute_at(mm.in(), x)
                .store_in(MemoryType::WMMAAccumulator)
                .update()
                .split(x, x, rxi, tile_x)
                .split(y, y, ryi, tile_y)
                .split(r.x, rro, rri, tile_r)

                .split(rro, rro, rroo, 4)
                .reorder({rri, rxi, ryi, x, y, rroo, rro})
                // .reorder({rri, rxi, ryi, x, y, rro})
                .unroll(rroo)
                .unroll(x)
                .unroll(y)
                .atomic()
                .vectorize(rri)
                .vectorize(rxi)
                .vectorize(ryi);

            B.in(mm).compute_at(mm, rroo).store_in(MemoryType::WMMAB)
                .vectorize(x)
                .vectorize(y)
                ;

            // initialization
            mm.split(x, x, rxi, tile_x)
                .split(y, y, ryi, tile_y)
                .vectorize(rxi)
                .vectorize(ryi)
                .unroll(y);

            mm.in()
                .split(x, x, xi, tile_x)
                .split(xi, xi, mmxi, tile_x)
                .split(y, y, yi, tile_y * 4)
                .split(yi, yi, mmyi, tile_y)
                .gpu_blocks(x, y)
                .reorder({mmxi, mmyi, xi, yi, x, y})
                .unroll(xi)
                .unroll(yi)
                .vectorize(mmxi)
                .vectorize(mmyi)
                ;
        } else if (schedule == 1) {
            // on-the-fly conversion

            A.compute_at(mm, rro)
                .store_in(MemoryType::GPUShared)
                // .store_in(MemoryType::Heap)
                .split(x, x, rxi, 2)
                .fuse(y, rxi, y)
                .gpu_lanes(y);
            B.compute_at(mm, rro)
                .store_in(MemoryType::GPUShared)
                // .store_in(MemoryType::Heap)
                .split(y, y, ryi, 2)
                .fuse(x, ryi, x)
                .gpu_lanes(x);

            // update
            mm.compute_at(mm.in(), x)
                .store_in(MemoryType::WMMAAccumulator)
                .update()
                .split(x, x, rxi, tile_x)
                .split(y, y, ryi, tile_y)
                .split(r.x, rro, rri, tile_r)
                .reorder({rri, rxi, ryi, rro, x, y})
                .atomic()
                .vectorize(rri)
                .vectorize(rxi)
                .vectorize(ryi);

            // initialization
            mm.split(x, x, rxi, tile_x)
                .split(y, y, ryi, tile_y)
                .vectorize(rxi)
                .vectorize(ryi);

            Var mmxi("mmxi"),
                mmyi("mmyi");
            mm.in()
                .split(x, x, mmxi, tile_x)
                .split(y, y, mmyi, tile_y)
                .gpu_blocks(x, y)
                .reorder({mmxi, mmyi, x, y})
                // .atomic()
                .vectorize(mmxi)
                .vectorize(mmyi);

        } else if (schedule == 2) {
            // naive
            A.compute_root().gpu_tile(x, y, rxi, ryi, tile_x, tile_y);
            B.compute_root().gpu_tile(x, y, rxi, ryi, tile_x, tile_y);

            // update
            mm.compute_at(mm.in(), x)
                .store_in(MemoryType::WMMAAccumulator)
                .update()
                .split(x, x, rxi, tile_x)
                .split(y, y, ryi, tile_y)
                .split(r.x, rro, rri, tile_r)
                .reorder({rri, rxi, ryi, rro, x, y})
                .atomic()
                .vectorize(rri)
                .vectorize(rxi)
                .vectorize(ryi);

            // initialization
            mm.split(x, x, rxi, tile_x)
                .split(y, y, ryi, tile_y)
                .vectorize(rxi)
                .vectorize(ryi);

            Var mmxi("mmxi"),
                mmyi("mmyi");
            mm.in()
                .split(x, x, mmxi, tile_x)
                .split(y, y, mmyi, tile_y)
                .gpu_blocks(x, y)
                .reorder({mmxi, mmyi, x, y})
                // .atomic()
                .vectorize(mmxi)
                .vectorize(mmyi);

        }
    } else {
        printf("Architecture not supported");
        exit(1);
    }

    Func result = mm.in();

    // result.compile_to_lowered_stmt("/tmp/matmul_flat_1x1.html", {A_input, B_input}, HTML, target);

    // test
    int row = 4096;
    int col = 4096;
    Buffer<float> a_buf(acc, row);
    fill_buffer_flat_one(a_buf, row, acc);
    A_input.set(a_buf);

    Buffer<float> b_buf(col, acc);
    fill_buffer_flat_one(b_buf, acc, col);
    B_input.set(b_buf);

    Buffer<float> out(col, row);
    auto time = Tools::benchmark(5, 5, [&]() {
        result.realize(out, target);
        if (use_gpu) {
            out.device_sync();
        }
    });

    if (use_gpu) {
        out.copy_to_host();
    }

    if (0) {
        int row = 4096;
        int col = 4096;

        for (int j = 0; j < row; ++j) {
            for (int i = 0; i < col; ++i) {
                float val = 0;
                for (int k = 0; k < acc; ++k) {
                    val += a_buf(k, j) * b_buf(i, k);
                }
                if (fabs(val - out(i, j)) > 0.001) {
                    std::cerr << "Invalid result at " << i << ", " << j << "\n"
                              << out(i, j) << " != " << val << "\n";
                    return false;
                }
            }
        }
    }

    std::cout << "Exec time: " << time << "\n";
    std::cout << "Success!\n";
    return true;
}

int main(int argc, char **argv) {
    freopen("/tmp/matmul_flat_1x1.log", "w", stderr);
    Target target("x86-64-linux-avx512_sapphirerapids");
    // Target target("x86-64-linux-cuda_capability_70");
    // Target target = get_target_from_environment().with_feature(Target::CUDA).with_feature(Target::CUDACapability70)
        // .with_feature(Target::Debug)
        // ;
    // Target target = get_jit_target_from_environment();
    std::cout << target;

    printf("Running AMX (bf16)\n");
    matmul_bf16(target);
    return 0;
}
