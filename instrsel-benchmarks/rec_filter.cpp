#include "Halide.h"
// #include "halide_benchmark.h"
// #include "halide_test_dirs.h"
#include <iomanip>
#include <iostream>

using namespace Halide;

bool matmul_bf16(Halide::Target target) {
    (void)target;

    // const int acc = 4096;
    // const int X_ACC = 1;
    // const int Y_ACC = 2;

    constexpr int delay_factor = 16;
    constexpr int width = 8192;
    constexpr int order = 2;

    Var x("x"), y("y"), n("n"), m("m"), mo("mo"), mi("mi"), mii("mii");
    ImageParam g(Float(16), 2, "g");
    // Buffer<float16_t, 2> A, B;
    ImageParam A(Float(16), 2, "A");
    ImageParam B(Float(16), 2, "B");
    RDom w(0, delay_factor, "w");

    Func f("f"), f_delay("f_delay");
    Func g_delay("g_delay");
    Func h("h");

    g_delay(mii, mi, mo, n) = g((mo * delay_factor + mi) * delay_factor + mii, n);

    h(mii, mi, mo, n) = cast<float>(0.f);
    h(mii, mi, mo, n) += cast<float>(A(mii, w)) * cast<float>(g_delay(w, mi, mo, n));

    f_delay(mi, mo, n) = h(mi, mo % delay_factor, mo / delay_factor, n);

    RDom r(0, 2,
           1, width / delay_factor,
           0, delay_factor,
           "r");
    f_delay(r.z, r.y, n) +=
        B(r.z, r.x) * f_delay(delay_factor - r.x - 1, r.y - 1, n);

    f(m, n) = f_delay(m % delay_factor, m / delay_factor, n);

    Func result = f;


    A.dim(0).set_bounds(0, delay_factor);
    A.dim(1).set_bounds(0, delay_factor);
    B.dim(0).set_bounds(0, delay_factor);
    B.dim(1).set_bounds(0, order);
    result.output_buffer().dim(0).set_bounds(0, width);
    result.output_buffer().dim(1).set_bounds(0, width);
    h.bound(mii, 0, delay_factor)
     .bound(mi, 0, delay_factor)
     .bound(mo, 0, width / delay_factor / delay_factor);
    f_delay.bound(mi, 0, delay_factor).bound(mo, 0, width / delay_factor);


    // A.in().store_root();
    // g_delay.compute_root();
    h.compute_at(h.in(), mo)
     .store_in(MemoryType::WMMAAccumulator)
     .reorder(mii, mi, mo, n)
     .vectorize(mi)
     .vectorize(mii)
     .update()
     .reorder(w, mii, mi, mo, n)
     .atomic()
     .vectorize(mi)
     .vectorize(mii)
     .vectorize(w);

    h.in()
     .compute_root()
     .reorder(mii, mi, mo, n)
     .gpu_blocks(n, mo)
     .vectorize(mii)
     .vectorize(mi)
     ;


    // f.split(m, mo, mi, delay_factor).gpu_blocks(n,mo)
    //  .gpu_threads(mi);

    // f_delay.compute_root();
    Var moo("moo"), moi("moi");
    f.gpu_blocks(n)
     .split(m, mo, mi, 32)
     .gpu_threads(mi);
    f_delay
     .compute_at(f, n)
     .fuse(mi, mo, m)
     .split(m, moo, moi, 32)
     .gpu_threads(moi)
    ;
    f_delay
     .update()
     .atomic(true)
     .gpu_threads(r.z)
     .unroll(r.x);

    // Issues
    // The load of h.in() to f_delay is over-complicated, it should just be a consecutive load followed by a consecutive store.
    // An error is thrown saying f_delay should have size 513 instead of 512.
    // Too many extra copying


    // Uncomment to check the asm
    result.compile_to_llvm_assembly("tiled_matmul_bf16.ll", {A, B, g}, target);
    // result.compile_to_assembly(Internal::get_test_tmp_dir() + "tiled_matmul.s", {A, B}, target);
    // result.compile_to_lowered_stmt("/tmp/rec-filter.html", {g}, HTML, target);

    std::cout << "Success!\n";
    return true;
}

int main(int argc, char **argv) {
    freopen("/tmp/rec-filter.log", "w", stderr);
    Target target("x86-64-linux-cuda-cuda_capability_70");

    matmul_bf16(target);
    return 0;
}
