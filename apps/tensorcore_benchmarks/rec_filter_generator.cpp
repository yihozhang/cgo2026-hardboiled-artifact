#include "Halide.h"
#include <cassert>
#include <stdio.h>

#include "common.h"

using namespace Halide;

constexpr int delay_factor = 16;
const int tile_width = 64;
class RecFilter : public Halide::Generator<RecFilter> {
public:
    // Generator Params
    GeneratorParam<Schedule> gpu_schedule{"gpu_schedule", Schedule::CUDA, {{"cuda_only", Schedule::CUDA}, {"tensorcore", Schedule::TensorCore}}};
    GeneratorParam<int> imgCol{"imgCol", 1024 * 1024};
    GeneratorParam<int> order{"order", 2};

    // Inputs
    Input<Buffer<float16_t>> g{"g", 2}; // inputs
    Input<Buffer<float>> a{"a", 1}; // coefficients
    Output<Buffer<float>> f{"f", 4};

    void generate() {
        /*---------------------------------*
        |  Compute A and B                 |
        *---------------------------------*/
        a1(m, n) = cast<float>(0);
        a1(0, 0) = cast<float>(1);

        // for (int o = 0; o < order; o++)
        //     for (int i = 1; i <= delay_factor; i++)
        //         for (int j = 1; j <= i; j++)
        //             if (j + o <= order)
        //                 a1(o, i) += a1(0, i - j) * cast<float>(a[j + o]);
        RDom r_init(
            1, delay_factor * tile_width,
            1, delay_factor * tile_width,
            0, order,
            "r_init"
        );
        r_init.where(r_init.x <= r_init.y);
        r_init.where(r_init.x + r_init.z <= order);
        
        a1(r_init.z, r_init.y) += a1(0, r_init.y - r_init.x) * cast<float>(a(r_init.x + r_init.z));

        wA = {0, delay_factor, 0, delay_factor, "wA"};
        wB = {0, delay_factor, 0, order, "wB"};
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
        f_initial(mii, m, n) = h(mii, m % delay_factor, m / delay_factor, n);

        // First intra-tile
        r = {0, order,
             0, delay_factor,
             // second arg is extent not upper bound
             1, tile_width - 1,
             "r"};
        fi(mii, mi, mo, n) = f_initial(mii, mo * tile_width + mi, n);
        fi(r.y, r.z, mo, n) += B(r.y, r.x) * fi(delay_factor - r.x - 1, r.z - 1, mo, n);

        // Inter-tile
        r_tail = {
            0, order,
            0, delay_factor,
            1, imgCol / tile_width / delay_factor - 1,
            "rt"
        };
        fct(mii, mo, n) = fi(mii, tile_width - 1, mo, n);
        fct(r_tail.y, r_tail.z, n) += a1(r_tail.x, (delay_factor - 1) * tile_width + r_tail.y) * fct(delay_factor - r_tail.x - 1, r_tail.z - 1, n);

        // Final intra-tile
        in_tile = {
            0, order,
            // 1, imgCol / tile_width / delay_factor - 1,
            "in_tile"
        };
        f(mii, mi, mo, n) = f_initial(mii, mo * tile_width + mi, n);
        // f_delay(mii, 0, in_tile.y, n) += a1(in_tile.x, mii) * fct(delay_factor - in_tile.x - 1, in_tile.y - 1, n);
        f(mii, 0, mo, n) += select(mo > 0, a1(in_tile.x, mii + 1) * fct(delay_factor - in_tile.x - 1, mo - 1, n), 0);
        f(r.y, r.z, mo, n) += B(r.y, r.x) * f(delay_factor - r.x - 1, r.z - 1, mo, n);
        
        // f_delay(mii, m, n) = h(mii, m % delay_factor, m / delay_factor, n);
        //
        // r = {0, order,
        //      0, delay_factor,
        //      // second arg is extent not upper bound
        //      1, imgCol / delay_factor - 1,
        //      "r"};
        // f_delay(r.y, r.z, n) +=
        //     B(r.y, r.x) * f_delay(delay_factor - r.x - 1, r.z - 1, n);

        // f = f_delay;
        // // Because in the update definition, r.y is a reduction variable,
        // // it is more difficult to do the reshaping as another step and
        // // compute f_delay at f, because the reduction domain won't be constrained
        // // to the range it is computed at.
        // // f(m, n) = f_delay(m % delay_factor, m / delay_factor, n);
    }

    void schedule() {
        bool debug = false;
        /*---------------------------------*
        | Bounds                           |
        *---------------------------------*/
        A.bound(m, 0, delay_factor).bound(n, 0, delay_factor);
        B.bound(m, 0, order).bound(n, 0, delay_factor);
        f_initial.bound(mii, 0, delay_factor).bound(m, 0, imgCol / delay_factor)
            .bound(n, 0, 2);  // stereo audio;
        f.bound(n, 0, 2);
        f.bound(mo, 0, imgCol / tile_width / delay_factor);
        f.bound(mi, 0, tile_width);
        f.bound(mii, 0, delay_factor);

        /*---------------------------------*
        |a1, A, B are small coeff matrices |
        *---------------------------------*/
        a1.compute_root().unroll(m);
        A.compute_root();
        B.compute_root();

        if (gpu_schedule == Schedule::CUDA) {
            h
                .compute_inline()
                .update()
                .unroll(w);
            f_delay
                .compute_root()
                .split(m, mo, mi, 16)
                .split(mo, moo, moi, 16)
                .reorder(mii, mi, moi, moo, n)
                .gpu_blocks(moo, n)
                .gpu_threads(mii, mi)
                .unroll(moi);
            f_delay
                .update()
                .atomic(true)
                .reorder(r.y, r.z, n)
                .gpu_blocks(n)
                .gpu_threads(r.y)
                .unroll(r.x);
        } else {
            // if (!debug) {
            //     h.compute_at(f_initial, moi);
            // } else {
            //     h.compute_at(f_initial, moi);
            // }
            h
                .compute_at(f_initial, moi)
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

            f_initial
                .compute_root()
                .split(m, mo, mi, 16)
                // computes (16 x 16) x 256 pixels in a block
                .split(mo, moo, moi, 256)
                .reorder(mii, mi, moi, moo, n)
                .gpu_blocks(moo, n)
                .vectorize(mi)
                .vectorize(mii)
                ;
            if (!debug) f_initial.split(moi, mo, moi, 8).unroll(moi);

            // TODO: split block into blocks of threads to increase occupancy
            fi
                .compute_root()
                .reorder(mii, mi, n, mo)
                .gpu_blocks(mo)
                .gpu_threads(mii, n)
                ;
            if (!debug) fi.unroll(mi, 8);
            fi
                .update()
                .atomic(true)
                .fuse(r.y, n, ri)
                .reorder(ri, r.z, mo)
                .gpu_blocks(mo)
                .gpu_threads(ri)
                ;

            fct
                .compute_root()
                .reorder(mii, n, mo)
                .gpu_blocks(mo)
                .gpu_threads(mii)
                .gpu_threads(n);
            fct
                .update()
                .atomic(true)
                .gpu_blocks(n)
                .gpu_threads(r_tail.y);

            f
                .compute_root()
                .fuse(mii, n, mii)
                .reorder(mii, mi, mo)
                .gpu_threads(mii)
                .gpu_blocks(mo)
                ;
            if (!debug) f.unroll(mi, 8);
            f
                .update(0)
                .atomic(true)
                .fuse(mii, n, mii)
                .gpu_threads(mii)
                .gpu_blocks(mo)
                .unroll(in_tile.x);
            f
                .update(1)
                .atomic(true)
                .fuse(r.y, n, ri)
                .gpu_threads(ri)
                .gpu_blocks(mo)
                .unroll(r.z, 8)
                ;
        }
    }

private:
    Var x{"x"}, y{"y"}, n{"n"}, m{"m"}, mo{"mo"}, mi{"mi"}, mii{"mii"};
    Var moo{"moo"}, moi{"moi"};
    RVar ro{"ro"}, ri{"ri"};

    Func a1{"a1"};
    Func A{"A"}, B{"B"};

    Func f_delay{"f_delay"}, f_initial{"f_initial"};
    Func g_delay{"g_delay"};
    Func h{"h"};
    Func fi{"fi"}, ft{"ft"}, fct{"fct"};

    RDom wA, wB;
    RDom w{0, delay_factor, "w"};
    RDom r;
    RDom r_tail;
    RDom in_tile;
};

HALIDE_REGISTER_GENERATOR(RecFilter, rec_filter)