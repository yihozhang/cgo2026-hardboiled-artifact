#include "Halide.h"
#include <cassert>
#include <stdio.h>

#include "common.h"

using namespace Halide;

constexpr int delay_factor = 16;
constexpr int order = 2;
// template<int order = 2>
class RecFilter : public Halide::Generator<RecFilter> {
public:
    // Generator Params
    GeneratorParam<Schedule> gpu_schedule{"gpu_schedule", Schedule::CUDA, {{"cuda_only", Schedule::CUDA}, {"tensorcore", Schedule::TensorCore}}};

    GeneratorParam<int> imgCol{"imgCol", 1024 * 1024 * 1024};

    // Inputs
    Input<Buffer<float16_t>> g{"g", 2};
    Input<float[order + 1]> a{"a"};
    Output<Buffer<float>> f{"f", 3};

    void generate() {
        /*---------------------------------*
        |  Compute A and B                 |
        *---------------------------------*/
        a1(m, n) = cast<float>(0);
        a1(0, 0) = cast<float>(1);

        for (int o = 0; o < order; o++) {
            for (int i = 1; i <= delay_factor; i++) {
                for (int j = 1; j <= i; j++) {
                    if (j + o <= order) {
                        a1(o, i) += a1(0, i - j) * cast<float>(a[j + o]);
                    }
                }
            }
        }

        A(m, n) = cast<float16_t>(0);
        A(wA.x, wA.y) = select(wA.y <= wA.x, cast<float16_t>(a1(0, wA.x - wA.y)), cast<float16_t>(0));
        B(n, m) = a1(m, n + 1);

        /*---------------------------------*
        | Computes the non-recursive part  |
        *---------------------------------*/
        g_delay(mii, mi, mo, n) = g((mo * delay_factor + mi) * delay_factor + mii, n);

        h(mii, mi, mo, n) = cast<float>(0.f);
        h(mii, mi, mo, n) += cast<float>(A(mii, w)) * cast<float>(g_delay(w, mi, mo, n));

        /*---------------------------------*
        | Computes the recursive part      |
        *---------------------------------*/
        f_delay(mii, m, n) = h(mii, m % delay_factor, m / delay_factor, n);

        r = {0, order,
             0, delay_factor,
             // second arg is extent not upper bound
             1, imgCol / delay_factor - 1,
             "r"};
        f_delay(r.y, r.z, n) +=
            B(r.y, r.x) * f_delay(delay_factor - r.x - 1, r.z - 1, n);

        f = f_delay;
        // Because in the update definition, r.y is a reduction variable,
        // it is more difficult to do the reshaping as another step and
        // compute f_delay at f, because the reduction domain won't be constrained
        // to the range it is computed at.
        // f(m, n) = f_delay(m % delay_factor, m / delay_factor, n);
    }

    void schedule() {
        /*---------------------------------*
        | Bounds                           |
        *---------------------------------*/
        A.bound(m, 0, delay_factor).bound(n, 0, delay_factor);
        B.bound(m, 0, order).bound(n, 0, delay_factor);
        f_delay.bound(mii, 0, delay_factor).bound(m, 0, imgCol / delay_factor)
            .bound(n, 0, 2);  // stereo audio;

        /*---------------------------------*
        |a1, A, B are small coeff matrices |
        *---------------------------------*/
        a1.compute_root();
        A.compute_root();
        B.compute_root();

        Var moo("moo"), moi("moi");

        h
            // Due to a bug in bound inference I cannot compute f_delay on-demand.
            .compute_at(f_delay, moi)
            // .compute_at(f_delay, moo)
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

        f_delay
            .compute_root()
            .split(m, mo, mi, 16)
            // computes (16 x 16) x 256 pixels in a block
            .split(mo, moo, moi, 256)
            .reorder(mii, mi, moi, moo, n)
            .gpu_blocks(moo, n)
            .vectorize(mi)
            .vectorize(mii)
            ;

        RVar fused("fused");
        RVar ro("ro"), ri("ri");
        f_delay
            .update()
            .atomic(true)
            .reorder(r.y, r.z, n)
            .gpu_blocks(n)
            .gpu_threads(r.y)
            .unroll(r.x)
            // .fuse(r.y, n, fused)
            // .split(r.z, ro, ri, 512)
            // .reorder(fused, ri, ro)
            // .reorder(fused, r.z)
            // .gpu_threads(fused)
            // .gpu_blocks(ro)
            .unroll(r.x)
            ;
    }

private:
    Var x{"x"}, y{"y"}, n{"n"}, m{"m"}, mo{"mo"}, mi{"mi"}, mii{"mii"};

    Func a1{"a1"};
    Func A{"A"}, B{"B"};

    Func f_delay{"f_delay"};
    Func g_delay{"g_delay"};
    Func h{"h"};

    RDom wA{0, delay_factor, 0, delay_factor, "wA"};
    RDom wB{0, delay_factor, 0, order, "wB"};

    RDom w{0, delay_factor, "w"};
    RDom r;
};

HALIDE_REGISTER_GENERATOR(RecFilter, rec_filter)