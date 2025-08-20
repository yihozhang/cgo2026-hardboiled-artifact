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

    GeneratorParam<int> imgRow{"imgRow", 4096};
    GeneratorParam<int> imgCol{"imgCol", 4096};

    // Inputs
    Input<Buffer<float16_t>> g{"g", 2};
    Input<float[order + 1]> a{"a"};
    Output<Buffer<float>> f{"f", 2};

    void generate() {
        if (gpu_schedule == Schedule::CUDA) {

            f(m, n) = cast<float>(0);

            for (int o = 0; o < order; o++) {
                Expr e = cast<float>(g(o, n));
                for (int i = 1; i <= o; i++) {
                    e = e + a[i] * f(o - i, n);
                }
                f(o, n) = e;
            }
            {
                r = {order, imgCol - order, "r"};
                Expr e = cast<float>(g(r.x, n));
                for (int i = 1; i <= order; i++) {
                    e = e + a[i] * f(r.x - i, n);
                }
                f(r.x, n) = e;
            }
        } else if (gpu_schedule == Schedule::TensorCore) {

            /*---------------------------------*
            |  Compute A and B                 |
            *---------------------------------*/
            // assert(a[0] == 0);
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

            g_delay(mii, mi, mo, n) = g((mo * delay_factor + mi) * delay_factor + mii, n);

            h(mii, mi, mo, n) = cast<float>(0.f);
            h(mii, mi, mo, n) += cast<float>(A(mii, w)) * cast<float>(g_delay(w, mi, mo, n));

            f_delay(mi, mo, n) = h(mi, mo % delay_factor, mo / delay_factor, n);

            r = {0, order,
                // second arg is extent not upper bound
                1, imgCol / delay_factor - 1,
                0, delay_factor,
                "r"};
            f_delay(r.z, r.y, n) +=
                B(r.z, r.x) * f_delay(delay_factor - r.x - 1, r.y - 1, n);

            f(m, n) = f_delay(m % delay_factor, m / delay_factor, n);
        } else {
            std::cerr << "Schedule not found\n";
            exit(1);
        }
    }

    void schedule() {
        if (gpu_schedule == Schedule::CUDA) {
            Var ni("ni"), no("no");
            // f.in()
            //  .split(n, no, ni, 32)
            //  .gpu_blocks(no)
            //  .gpu_threads(ni);
            f//.compute_at(f.in(), ni)
             .split(n, no, ni, 32)
             .gpu_blocks(no)
             .gpu_threads(ni);
            for (int o = 0; o <= order; o++) {
                f.update(o)
                 .split(n, no, ni, 32)
                 .gpu_blocks(no)
                 .gpu_threads(ni);;
            }
        } else {
            A.bound(m, 0, delay_factor).bound(n, 0, delay_factor);
            B.bound(m, 0, order).bound(n, 0, delay_factor);
            f.bound(m, 0, imgCol);
            f.bound(n, 0, imgRow);
            h.bound(mii, 0, delay_factor)
                .bound(mi, 0, delay_factor)
                .bound(mo, 0, imgCol / delay_factor / delay_factor);
            f_delay.bound(mi, 0, delay_factor).bound(mo, 0, imgCol / delay_factor);
            
            a1.compute_root();
            A.compute_root();
            B.compute_root();

            Var moo("moo"), moi("moi");

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
                // .compute_root()
                .compute_at(f_delay, moo)
                .reorder(mii, mi, mo, n)
                // .gpu_blocks(n, mo)
                .vectorize(mii)
                .vectorize(mi);

            // f.split(m, mo, mi, delay_factor).gpu_blocks(n,mo)
            //  .gpu_threads(mi);

            // f_delay.compute_root();
            f.gpu_blocks(n)
                .split(m, mo, mi, 32)
                .gpu_threads(mi);
            f_delay
                .compute_at(f, n)
                .fuse(mi, mo, m)
                .split(m, moo, moi, 32)
                .gpu_threads(moi);
            f_delay
                .update()
                .atomic(true)
                .gpu_threads(r.z)
                .unroll(r.x);
        }
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