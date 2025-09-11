#include "Halide.h"
#include <cassert>
#include <stdio.h>

#include "common.h"

using namespace Halide;

class RecFilter : public Halide::Generator<RecFilter> {
public:
    // Generator Params
    GeneratorParam<Schedule> gpu_schedule{"gpu_schedule", Schedule::CUDA, {{"cuda_only", Schedule::CUDA}, {"tensorcore", Schedule::TensorCore}}};

    // Input: 2-channel float16 audio
    Input<Buffer<float16_t>> g{"g", 2};

    // Filter coefficients
    Input<uint16_t> a1_bits{"a1_bits"}, a2_bits{"a2_bits"};

    // Filter impulse response (precomputed because it doesn't depend on the input)
    Input<Buffer<float16_t>> impulse{"impulse", 2};

    // Output
    Output<Buffer<float16_t>> output{"output", 2};

    void generate() {
        // Tunable tiling factor. 1024 seems to be near optimal for both schedules.
        int tile_width = 1024;

        Expr a1 = reinterpret<float16_t>(a1_bits);
        Expr a2 = reinterpret<float16_t>(a2_bits);

        // This is a recursive filter. The recurrence relation is:
        // output[x] = input[x] + a1 * output[x-1] + a2 * output[x-2]

        // A naive implementation is serial. There are lots of ways to factor a
        // recursive filter to extract some parallelism. Here's how to think
        // about all of them:

        // This whole thing is linear. No matter what tricks we play, everything
        // we ever compute is going to be linearly predictable from some subset
        // of the previously seen inputs and previously produced
        // outputs. Because it's linear we can characterize it in terms of the
        // response to standard basis vectors. i.e. how does the system evolve
        // over time if a single input, or a single previous output is 1, and
        // everything else is zero.

        // If a single input is 1 and all other inputs and all previous outputs
        // are zero, you get the impulse response of the filter starting at the
        // 1. That's the definition of the impulse response.

        // If the immediately previous output is 1, outputs before that are
        // zero, and all inputs are zero, you also get the impulse response of
        // the filter starting at the 1. This is because in our recurrence,
        // there's no coefficient on the input.

        // It's more complicated if the output from time t - 2 is 1, and the
        // output at t-1 and all inputs are zero. If we just took the impulse
        // response starting at the 1 (at t-2), then the output at t-1 would
        // have been the coefficient a1, not zero. To make it zero for that
        // output we need to subtract the impulse response centered at that next
        // output scaled by a1. So the system applied to the basis vector which
        // is [1 0] for the previous two outputs produces the impulse response
        // convolved with [1 -a1]. These two impulse responses vary with the
        // coefficients, not the input, so we precompute them once outside the
        // kernel and accept them as an additional input buffer. Really Halide
        // needs a .super_root() mechanism to lift compute outside of kernels
        // entirely.

        // With that out of the way, how are we actually parallelizing this
        // apparently serial filter? We want outer-loop parallelism (tiles) and
        // inner-loop parallelism (warp lanes). To get our outer loop
        // parallelism over tiles, we use GPU-efficient Recursive Filtering by
        // Hoppe et al. This turns a recursive filter into performing the filter
        // within each tile in isolation, and then adding a correction factor
        // using the end of the previous tile. If the filter decays slowly this
        // correction factor still needs to be computed serially across the
        // tiles, but it's O(1) work per tile.

        // This gives you a good parallel axis for GPU blocks. You can map GPU
        // threads to this too if you want, but it results in wraps doing large
        // strided loads and not much parallelism in general. It's possible to
        // do better. To get inner-loop parallelism we'll use
        // scattered-lookahead interpolation (SLA) from "Pipeline interleaving
        // and parallelism in recursive digital filters" by K K Parhi et
        // al. This turns a dense IIR into an FIR filter followed by a strided
        // IIR filter, allowing you to parallelize within one stride. A nice
        // thing about SLA is that it also makes the filter more numerically
        // stable.

        Var m{"m"}, n{"n"}, mi{"mi"}, mo{"mo"}, o{"o"}, mii{"mii"};

        // Stage 1: scan within each tile, pretending everything outside the
        // tile is zero. This is parallel over tiles.

        // Within that tile we'll use scattered lookahead interpolation, as
        // described above. We need to precompute a small amount of stuff to do
        // SLA, giving us our dense FIR and our strided IIR.

        // Figure out the dilated convolution kernel in closed form on host
        std::vector<Expr> kernel_ = {Expr(1.0f)};
        Expr a1_ = cast<double>(a1), a2_ = cast<double>(a2);

        int dilation = 1;
        // The optimal amount of dilation could in principle vary per schedule,
        // but it seems to be 8 for both.
        int target_dilation = gpu_schedule == Schedule::CUDA ? 8 : 8;
        while (dilation < target_dilation) {
            auto convolve = [&](std::vector<Expr> &a, std::vector<Expr> b, int dilation) {
                size_t b_dilated_size = (b.size() - 1) * dilation + 1;
                std::vector<Expr> result(a.size() + b_dilated_size - 1);
                for (size_t i = 0; i < result.size(); i++) {
                    Expr e = 0.f;
                    for (size_t k = 0; k < b.size(); k++) {
                        ptrdiff_t j = i - dilation * k;
                        if (j >= 0 && j < (ptrdiff_t)a.size()) {
                            e += a[j] * b[k];
                        }
                    }
                    result[i] = e;
                }
                return result;
            };
            kernel_ = convolve(kernel_, std::vector<Expr>{Expr(1.0), a1_, -a2_}, dilation);
            a1_ = a1_ * a1_ + 2 * a2_;
            a2_ = -a2_ * a2_;
            dilation *= 2;
        }
        a1_ = cast<float16_t>(a1_);
        a2_ = cast<float16_t>(a2_);
        kernel_.push_back(0.f);
        std::reverse(kernel_.begin(), kernel_.end());

        Func kernel{"kernel"};
        kernel(mi) = cast<float16_t>(mux(mi, kernel_));
        kernel.compute_root().unroll(mi);

        // We are now ready to perform the FIR
        Func g_tiled("g_tiled");
        g_tiled(mi, mo, n) = g(mo * tile_width + mi, n);

        Func g_padded("g_padded");
        g_padded(mi, mo, n) = select(mi >= 0, likely(g_tiled(max(0, mi), mo, n)), cast<float16_t>(0.f));

        Func g_convolved("g_convolved");
        g_convolved(mi, mo, n) = 0.f;
        RDom r_conv(0, dilation * 2);
        g_convolved(mi, mo, n) += cast<float>(kernel(r_conv)) * g_padded(mi - 2 * dilation + 1 + r_conv, mo, n);

        // Next we perform the strided/dilated IIR
        Func fi_dilated{"fi_dilated"};
        fi_dilated(mii, mi, mo, n) = undef<float>();
        RDom r_intra{0, tile_width / dilation, "r_intra"};
        fi_dilated(mii, r_intra, mo, n) =
            (g_convolved(r_intra * dilation + mii, mo, n) +
             select(r_intra > 0, a1_ * fi_dilated(mii, r_intra - 1, mo, n), 0.f) +
             select(r_intra > 1, likely(a2_ * fi_dilated(mii, r_intra - 2, mo, n)), 0.f));

        // Undo the sub-tiling
        Func fi{"fi"};
        fi(mi, mo, n) = fi_dilated(mi % dilation, mi / dilation, mo, n);

        // Stage 2 of Hoppe et al.: A recursive scan across the last two values
        // in each tile, to correct the fact that we were pretending values
        // outside the tile were zero. This is serial, but it only touches the
        // last two values in each tile, so it's not so bad.
        Func fct{"fct"};
        fct(o, m, n) = select(m < 0, 0.f, fi(tile_width - 2 + o, max(0, m), n));
        RDom r_tail{0, 2, 0, output.width() / tile_width, "r_tail"};
        // Doing this recursive sweep is a no-op if the filter decays entirely in
        // the span of one tile, which most numerically-stable filters will, if
        // the tiles are large. We skip it in that case.
        r_tail.where(impulse(tile_width, 0) > 0);
        fct(r_tail.x, r_tail.y, n) +=
            (impulse(tile_width - 1 + r_tail.x, 1) * fct(0, r_tail.y - 1, n) +
             impulse(tile_width - 1 + r_tail.x, 0) * fct(1, r_tail.y - 1, n));

        // Stage 3: Finally, we correct the other values in each tile using the
        // last two values of the previous tile.
        Func f{"f"};
        f(mi, mo, n) =
            // Intra-tile contribution
            cast<float16_t>(fi(mi, mo, n) +
                            // Contribution from 2nd-last value in previous tile
                            impulse(mi + 1, 1) * fct(0, mo - 1, n) +
                            // Contribution from the last value in the previous tile
                            impulse(mi + 1, 0) * fct(1, mo - 1, n));

        // Untile the result
        output(m, n) = f(m % tile_width, m / tile_width, n);

        // Schedule
        Var moo{"moo"}, moi{"moi"}, mio{"mio"}, z{"z"};
        RVar ri{"ri"}, ro{"ro"}, rio{"rio"}, rii{"rii"};
        fi_dilated.compute_root();
        fi_dilated.update()
            .gpu_blocks(mo, n)
            .split(mo, moo, moi, std::max(1, 32 / dilation))
            .fuse(mii, moi, z)
            .gpu_threads(z)
            // Halide really doesn't want to keep the last two outputs in
            // registers across the loop back edge. It wants to reload after
            // storing. We can approximate the correct behavior by just
            // unrolling a lot.
            .split(r_intra, ro, ri, 32)
            .unroll(ri);

        if (gpu_schedule == Schedule::CUDA) {
            g_convolved
                .compute_at(fi_dilated, ri)
                .update()
                .unroll(r_conv);

        } else {
            fi_dilated.update().reorder(ri, ro, z);

            g_convolved.in()
                .compute_at(fi_dilated, moo)
                .split(mi, mio, mii, 256)
                .unroll(mio)
                .vectorize(mii);

            // The dense FIR is going to use mma instructions. The FIR kernel
            // will automatically be shuffled into Toeplitz form for us.
            g_convolved.compute_at(g_convolved.in(), mo)
                .store_in(MemoryType::WMMAAccumulator)
                .vectorize(mi, 256)
                .unroll(mi)
                .unroll(mo)
                .update()
                .vectorize(mi, 256)
                .unroll(mi)
                .unroll(mo)
                .atomic()
                .vectorize(r_conv, 8)
                .unroll(r_conv);

            g_padded.compute_at(g_convolved, mo)
                .vectorize(mi, 4)
                .split(mi, mio, mii, 32)
                .gpu_lanes(mii)
                .unroll(mio);
        }

        fct.compute_root()
            .gpu_tile(m, mo, mi, 32)
            .gpu_blocks(n)
            .vectorize(o);

        fct.update(0)
            .gpu_blocks(n)
            .unroll(r_tail.x)
            .unroll(r_tail.y, 4);

        output.compute_root()
            .gpu_tile(m, mo, mi, 128, TailStrategy::RoundUp)
            .vectorize(mi, 4)
            .gpu_blocks(n);

        // Simplify some indexing
        output.dim(0).set_bounds(0, (output.dim(0).extent() >> 16) << 16);
    }
};

HALIDE_REGISTER_GENERATOR(RecFilter, rec_filter)
