#include "Halide.h"
#include "halide_benchmark.h"
#include "halide_test_dirs.h"

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

template<typename IntT>
void fill_buffer_a(Buffer<IntT> &buf, int row, int acc) {
    for (int iy = 0; iy < row; iy++) {
        for (int ix = 0; ix < acc; ix++) {
            buf(ix, iy) = rand() % 256 + std::numeric_limits<IntT>::min();
        }
    }
}

template<typename IntT>
void fill_buffer_b(Buffer<IntT> &buf, int col, int acc) {
    for (int iy = 0; iy < acc / 4; iy++) {
        for (int ix = 0; ix < col; ix++) {
            for (int ik = 0; ik < 4; ++ik) {
                buf(ik, ix, iy) = rand() % 256 + std::numeric_limits<IntT>::min();
            }
        }
    }
}

bool equal_eps(float lhs, float rhs, float eps) {
    return std::abs(lhs - rhs) < eps;
}

struct make_uint_t {
    template<typename... Args>
    Type operator()(Args &&...args) const {
        return UInt(static_cast<Args &&>(args)...);
    }
};

struct make_int_t {
    template<typename... Args>
    Type operator()(Args &&...args) const {
        return Int(static_cast<Args &&>(args)...);
    }
};

template<typename T>
void print_mat(const Buffer<T> &buf, int rows, int cols) {
    using cast_T = std::conditional_t<std::is_integral_v<T>, int, T>;
    for (int j = 0; j != rows; ++j) {
        for (int i = 0; i != cols; ++i) {
            std::cout << static_cast<cast_T>(buf(i, j)) << " ";
        }
        std::cout << std::endl;
    }
}

template<typename T>
void print_mat_rhs(const Buffer<T> &buf, int rows, int cols) {
    using cast_T = std::conditional_t<std::is_integral_v<T>, int, T>;
    for (unsigned long int j = 0; j != (rows / (4 / sizeof(T))); ++j) {
        for (int k = 0; k != (4 / sizeof(T)); ++k) {
            for (int i = 0; i != cols; ++i) {
                std::cout << static_cast<cast_T>(buf(k, i, j)) << " ";
            }

            std::cout << std::endl;
        }
    }
}

bool matmul_bf16(Halide::Target target) {
    (void)target;

    const int acc = 4096;

    Var x("x"), y("y");
    ImageParam A(BFloat(16), 2, "lhs");
    ImageParam B(BFloat(16), 3, "rhs");

    B.dim(1).set_stride(2);

    RDom r(0, acc, "acc");

    Func mm("matmul");
    mm(x, y) = cast<float>(0);
    mm(x, y) += cast<float>(cast<float>(A(r.x, y))) * cast<float>(B(r.x % 2, x, r.x / 2));

    int tile_x = 16;
    int tile_y = 16;
    int tile_r = 32;
    Var rxi("rxi"), ryi("ryi");
    RVar rri("rri"), rro("rro");

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

    // Uncomment to check the asm
    // result.compile_to_llvm_assembly(Internal::get_test_tmp_dir() + "tiled_matmul_bf16.ll", {A, B}, target);
    // result.compile_to_assembly(Internal::get_test_tmp_dir() + "tiled_matmul.s", {A, B}, target);
    // result.compile_to_lowered_stmt("/tmp/matmul_vnni_1x1.html", {A, B}, HTML, target);

    int row = 64, col=64;
    {
        Buffer<bfloat16_t> A_buf(acc, row);
        Buffer<bfloat16_t> B_buf(2, col, acc / 2);
        fill_buffer_a_bf16(A_buf, row, acc);
        A.set(A_buf);
        fill_buffer_b_bf16(B_buf, col, acc);
        B.set(B_buf);

        Buffer<float> out(col, row);

        // Uncomment to check the asm
        // result.compile_to_llvm_assembly(Internal::get_test_tmp_dir() + "tiled_matmul_bf16.ll", {A_buf, B_buf}, target);
        // result.compile_to_assembly(Internal::get_test_tmp_dir() + "tiled_matmul.s", {A_buf, B_buf}, target);

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
                    val += static_cast<float>(A_buf(k, j)) * static_cast<float>(B_buf(k % 2, i, k / 2));
                }
                if (!equal_eps(val, out(i, j), 1.f)) {
                // if (!equal_eps(val, out(i, j), 0.03f)) {
                    std::cerr << "Invalid result at " << i << ", " << j << "\n"
                            << out(i, j) << " != " << val << "\n"
                            << "Matrix dims: " << row << "x" << col << "x" << acc << "\nTile dims: " << tile_x << "x" << tile_y << "x" << tile_r << "\n";
                    return false;
                }
            }
        }
    }

    return true;
}

int main(int argc, char **argv) {
    freopen("/tmp/matmul_preload_vnni.log", "w", stderr);
    Target target("x86-64-linux-avx512_sapphirerapids");

    printf("Running AMX (bf16)\n");
    matmul_bf16(target);
    return 0;
}
