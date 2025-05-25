#include "Halide.h"
#include "halide_benchmark.h"
#include "halide_test_dirs.h"
#include "matrix_generator.h"

#include <iomanip>
#include <iostream>

using namespace Halide;

void fill_buffer_a_bf16(Buffer<bfloat16_t> &buf, int row, int acc) {
    for (int iy = 0; iy < row; ++iy) {
        for (int ix = 0; ix < acc; ++ix) {
            // value between 0 and 100
            bfloat16_t val = bfloat16_t(((float)rand() / (float)(RAND_MAX)) * 100.f - 50);
            // bfloat16_t val = bfloat16_t(1);
            buf(ix, iy) = val;
        }
    }
}

void fill_buffer_b_bf16(Buffer<bfloat16_t> &buf, int col, int acc) {
    for (int iy = 0; iy < acc / 2; ++iy) {
        for (int ix = 0; ix < col; ++ix) {
            for (int ik = 0; ik < 2; ++ik) {
                bfloat16_t val = bfloat16_t(((float)rand() / (float)(RAND_MAX)) * 100.f - 50);
                // bfloat16_t val = bfloat16_t(1);
                buf(ik, ix, iy) = val;
            }
        }
    }
}

bool matmul_bf16(Halide::Target target) {
    (void)target;

    Var x("x"), y("y");

    ImageParam A(BFloat(16), 2);
    ImageParam B(BFloat(16), 2);
    
    // Buffer<bfloat16_t> A(32, 16);
    // Buffer<bfloat16_t> B(16, 32);
    // Func A(16, 32, "lhs"), B(16, 16, "rhs");

    RDom r(0, 32, "acc");

    Func mm("matmul");

    // if (target.has_feature(Target::AVX512_SapphireRapids)) {
        mm(y, x) = 0.f;
        mm(y, x) += cast<float>(A(r, x)) * cast<float>(B(y, r));

        // A.in().bound(_0, 0, 16).bound(_1, 0, 32);
        // B.in().bound(_0, 0, 16).bound(_1, 0, 16);
        // mm.bound(y, 0, 16).bound(x, 0, 32);
        mm.in().bound(y, 0, 16).bound(x, 0, 16);
        A.dim(0).set_min(0).dim(1).set_stride(32).set_min(0);
        B.dim(0).set_min(0).dim(1).set_stride(16);//.set_min(0);

        mm.compute_at(mm.in(), x)
            .store_in(MemoryType::AMXTile)
            .vectorize(x, 16)
            .vectorize(y, 16)
            // .vectorize(x)
            // .vectorize(y)
            .update()
            .atomic()
            .vectorize(r, 32)
            .vectorize(y, 16)
            .vectorize(x, 16)
            // .vectorize(x)
            // .vectorize(y)
            // .vectorize(r)
            ;
        mm.in()
        .vectorize(x, 16)
        .vectorize(y, 16)
        // .vectorize(x)
        // .vectorize(y)
        ;
    // }

    Func result = mm.in();

    // result.compile_to_lowered_stmt("/tmp/matmul_flat_1x1.html", {A, B}, HTML, target);
    {
        int row = 16;
        int col = 16;
        int acc = 32;
        Buffer<bfloat16_t> A_buf(acc, row);
        Buffer<bfloat16_t> B_buf(col, acc);
        fill_buffer_a_bf16(A_buf, row, acc);
        A.set(A_buf);
        fill_buffer_a_bf16(B_buf, acc, col);
        B.set(B_buf);

        Buffer<float> out(col, row);
        result.realize(out);

        // uncomment to check the matrices
        // std::cout << "Matrix A_buf\n";
        // print_mat(A_buf, row, acc);
        // std::cout << "Matrix B_buf\n";
        // print_mat_rhs(B_buf, acc, col);

        // std::cout << "result\n";
        // print_mat(out, row, col);

        for (int j = 0; j < row; ++j) {
            for (int i = 0; i < col; ++i) {
                float val = 0.f;
                for (int k = 0; k < acc; ++k) {
                    val += static_cast<float>(A_buf(k, j)) * static_cast<float>(B_buf(i, k));
                }
                if (abs(val - out(i, j) > 0.1f)) {
                // if (!equal_eps(val, out(i, j), 0.03f)) {
                    std::cerr << "Invalid result at " << i << ", " << j << "\n"
                            << out(i, j) << " != " << val << "\n";
                    return false;
                }
            }
        }
    }

    // std::cout << "Exec time: " << time << "\n";
    std::cout << "Success!\n";
    return true;
}

int main(int argc, char **argv) {
    freopen("/tmp/example.log", "w", stderr);
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
