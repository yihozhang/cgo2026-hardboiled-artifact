#include "Halide.h"
#include <stdio.h>

#include "Halide.h"
#include "common.h"

using namespace Halide;

class Denoise : public Halide::Generator<Denoise> {
public:
    GeneratorParam<Schedule> gpu_schedule{
        "gpu_schedule", Schedule::CUDA,         //
        {                                       //
         {"cuda_only", Schedule::CUDA},         //
         {"tensorcore", Schedule::TensorCore}}  //
    };

    Input<Buffer<float16_t, 3>> input{"input"};
    Input<float> strength{"strength"};
    Output<Buffer<float16_t, 3>> output{"output"};

    void generate() {
        Var x("x"), y("y"), xi("xi"), yi("yi"), c("c");

        // Break the image into overlapping 16x16 tiles
        Func tiled("tiled");
        tiled(xi, yi, x, y, c) = input(x * 8 + xi, y * 8 + yi, c);

        // Take a DCT of each tile. We'll use both a direct DCT, computing the
        // matrices at compile-time, and a fast DCT. CUDA cores can run either,
        // but tensorcores really need to use the brute-force one.

        // Compute Hann windows and the brute-force DCT and IDCT matrices
        Buffer<float16_t> IDCT(16, 16), DCT(16, 16);
        float hann[16], inverse_hann[16];
        for (int i = 0; i < 16; i++) {
            hann[i] = 0.5 * (1 - cos(2 * M_PI * (i + 1) / 17));
        }
        // Because of the overlap-add, the inverse window is a little more complicated.
        // Each point is going to be the sum of four weighted values, where the sum of
        // the weights for pixel i, j is:
        // (hann[i] + hann[(i + 8) & 16]) * (hann[j] + hann[(j + 8) % 16])
        for (int i = 0; i < 16; i++) {
            inverse_hann[i] = 1 / (hann[i] + hann[(i + 8) % 16]);
        }

        for (int j = 0; j < 16; j++) {  // Row
            float alpha = (j == 0) ? (1.0f / sqrt(16.0f)) : sqrt(2.0f / 16.0f);
            for (int i = 0; i < 16; i++) {  // Col
                float f = alpha * cos((M_PI / 16) * (i + 0.5f) * j);
                DCT(i, j) = float16_t(f * hann[i]);
                IDCT(j, i) = float16_t(f * inverse_hann[i]);
            }
        }

        // Now we want to know how much energy to expect in each bin after
        // taking the transform, so that we can set thresholds. Consider
        // zero-mean Gaussian random noise with variance 1. After taking a DCT
        // the expected distribution per bin is still a Gaussian, with a
        // variance equal to the sum of the squares of that row of the
        // matrix. If it were the pure DCT matrix that'd be one, but we messed
        // it up with the windowing, so we need to compute it.
        Buffer<float> noise_floor(16);
        for (int j = 0; j < 16; j++) {
            float n = 0;
            for (int i = 0; i < 16; i++) {
                float v = float(DCT(i, j));
                n += v * v;
            }
            noise_floor(j) = std::sqrt(n);
        }

        // A fast DCT implementation derived from
        // https://github.com/norishigefukushima/dct_simd/blob/master/dct/dct16x16_simd.cpp
        // Modified to incorporate the Hann windows.
        auto fast_DCT = [&](int dim, Func f, Func in, bool forwards = true) {
            std::vector<Var> vars{xi, yi, x, y, c};
            Func scratch("scratch");
            auto lhs = [&](int i) {
                if (dim == 0) {
                    return scratch(i, yi, x, y, c);
                } else {
                    return scratch(xi, i, x, y, c);
                }
            };

            auto src = [&](int i) {
                if (dim == 0) {
                    return cast<float>(in(i, yi, x, y, c));
                } else {
                    return cast<float>(in(xi, i, x, y, c));
                }
            };

            scratch(xi, yi, x, y, c) = undef<float>();
            if (forwards) {
                lhs(0x00) = hann[0] * (src(0) + src(15));
                lhs(0x01) = hann[1] * (src(1) + src(14));
                lhs(0x02) = hann[2] * (src(2) + src(13));
                lhs(0x03) = hann[3] * (src(3) + src(12));
                lhs(0x04) = hann[4] * (src(4) + src(11));
                lhs(0x05) = hann[5] * (src(5) + src(10));
                lhs(0x06) = hann[6] * (src(6) + src(9));
                lhs(0x07) = hann[7] * (src(7) + src(8));
                lhs(0x08) = hann[0] * (src(0) - src(15));
                lhs(0x09) = hann[1] * (src(1) - src(14));
                lhs(0x0a) = hann[2] * (src(2) - src(13));
                lhs(0x0b) = hann[3] * (src(3) - src(12));
                lhs(0x0c) = hann[4] * (src(4) - src(11));
                lhs(0x0d) = hann[5] * (src(5) - src(10));
                lhs(0x0e) = hann[6] * (src(6) - src(9));
                lhs(0x0f) = hann[7] * (src(7) - src(8));

                lhs(0x10) = lhs(0x00) + lhs(0x07);
                lhs(0x11) = lhs(0x01) + lhs(0x06);
                lhs(0x12) = lhs(0x02) + lhs(0x05);
                lhs(0x13) = lhs(0x03) + lhs(0x04);
                lhs(0x14) = lhs(0x00) - lhs(0x07);
                lhs(0x15) = lhs(0x01) - lhs(0x06);
                lhs(0x16) = lhs(0x02) - lhs(0x05);
                lhs(0x17) = lhs(0x03) - lhs(0x04);
                lhs(0x18) = lhs(0x10) + lhs(0x13);
                lhs(0x19) = lhs(0x11) + lhs(0x12);
                lhs(0x1a) = lhs(0x10) - lhs(0x13);
                lhs(0x1b) = lhs(0x11) - lhs(0x12);

                lhs(0x1c) = 1.38703984532215f * lhs(0x14) + 0.275899379282943f * lhs(0x17);
                lhs(0x1d) = 1.17587560241936f * lhs(0x15) + 0.785694958387102f * lhs(0x16);
                lhs(0x1e) = -0.785694958387102f * lhs(0x15) + 1.17587560241936f * lhs(0x16);
                lhs(0x1f) = 0.275899379282943f * lhs(0x14) - 1.38703984532215f * lhs(0x17);
                lhs(0x20) = 0.25f * (lhs(0x1c) - lhs(0x1d));
                lhs(0x21) = 0.25f * (lhs(0x1e) - lhs(0x1f));
                lhs(0x22) = 1.40740373752638f * lhs(0x08) + 0.138617169199091f * lhs(0x0f);
                lhs(0x23) = 1.35331800117435f * lhs(0x09) + 0.410524527522357f * lhs(0x0e);
                lhs(0x24) = 1.24722501298667f * lhs(0x0a) + 0.666655658477747f * lhs(0x0d);
                lhs(0x25) = 1.09320186700176f * lhs(0x0b) + 0.897167586342636f * lhs(0x0c);
                lhs(0x26) = -0.897167586342636f * lhs(0x0b) + 1.09320186700176f * lhs(0x0c);
                lhs(0x27) = 0.666655658477747f * lhs(0x0a) - 1.24722501298667f * lhs(0x0d);
                lhs(0x28) = -0.410524527522357f * lhs(0x09) + 1.35331800117435f * lhs(0x0e);
                lhs(0x29) = 0.138617169199091f * lhs(0x08) - 1.40740373752638f * lhs(0x0f);
                lhs(0x2a) = lhs(0x22) + lhs(0x25);
                lhs(0x2b) = lhs(0x23) + lhs(0x24);
                lhs(0x2c) = lhs(0x22) - lhs(0x25);
                lhs(0x2d) = lhs(0x23) - lhs(0x24);
                lhs(0x2e) = 0.25f * (lhs(0x2a) - lhs(0x2b));
                lhs(0x2f) = 0.326640741219094f * lhs(0x2c) + 0.135299025036549f * lhs(0x2d);
                lhs(0x30) = 0.135299025036549f * lhs(0x2c) - 0.326640741219094f * lhs(0x2d);
                lhs(0x31) = lhs(0x26) + lhs(0x29);
                lhs(0x32) = lhs(0x27) + lhs(0x28);
                lhs(0x33) = lhs(0x26) - lhs(0x29);
                lhs(0x34) = lhs(0x27) - lhs(0x28);
                lhs(0x35) = 0.25f * (lhs(0x31) - lhs(0x32));
                lhs(0x36) = 0.326640741219094f * lhs(0x33) + 0.135299025036549f * lhs(0x34);
                lhs(0x37) = 0.135299025036549f * lhs(0x33) - 0.326640741219094f * lhs(0x34);

                lhs(0) = 0.25f * (lhs(0x18) + lhs(0x19));
                lhs(1) = 0.25f * (lhs(0x2a) + lhs(0x2b));
                lhs(2) = 0.25f * (lhs(0x1c) + lhs(0x1d));
                lhs(3) = 0.707106781186547f * (lhs(0x2f) - lhs(0x37));
                lhs(4) = 0.326640741219094f * lhs(0x1a) + 0.135299025036549f * lhs(0x1b);
                lhs(5) = 0.707106781186547f * (lhs(0x2f) + lhs(0x37));
                lhs(6) = 0.707106781186547f * (lhs(0x20) - lhs(0x21));
                lhs(7) = 0.707106781186547f * (lhs(0x2e) + lhs(0x35));
                lhs(8) = 0.25f * (lhs(0x18) - lhs(0x19));
                lhs(9) = 0.707106781186547f * (lhs(0x2e) - lhs(0x35));
                lhs(10) = 0.707106781186547f * (lhs(0x20) + lhs(0x21));
                lhs(11) = 0.707106781186547f * (lhs(0x30) - lhs(0x36));
                lhs(12) = 0.135299025036549f * lhs(0x1a) - 0.326640741219094f * lhs(0x1b);
                lhs(13) = 0.707106781186547f * (lhs(0x30) + lhs(0x36));
                lhs(14) = 0.25f * (lhs(0x1e) + lhs(0x1f));
                lhs(15) = 0.25f * (lhs(0x31) + lhs(0x32));
            } else {

                lhs(0x00) = 1.4142135623731f * src(0);
                lhs(0x01) = 1.40740373752638f * src(1) + 0.138617169199091f * src(15);
                lhs(0x02) = 1.38703984532215f * src(2) + 0.275899379282943f * src(14);
                lhs(0x03) = 1.35331800117435f * src(3) + 0.410524527522357f * src(13);
                lhs(0x04) = 1.30656296487638f * src(4) + 0.541196100146197f * src(12);
                lhs(0x05) = 1.24722501298667f * src(5) + 0.666655658477747f * src(11);
                lhs(0x06) = 1.17587560241936f * src(6) + 0.785694958387102f * src(10);
                lhs(0x07) = 1.09320186700176f * src(7) + 0.897167586342636f * src(9);
                lhs(0x08) = 1.4142135623731f * src(8);
                lhs(0x09) = -0.897167586342636f * src(7) + 1.09320186700176f * src(9);
                lhs(0x0a) = 0.785694958387102f * src(6) - 1.17587560241936f * src(10);
                lhs(0x0b) = -0.666655658477747f * src(5) + 1.24722501298667f * src(11);
                lhs(0x0c) = 0.541196100146197f * src(4) - 1.30656296487638f * src(12);
                lhs(0x0d) = -0.410524527522357f * src(3) + 1.35331800117435f * src(13);
                lhs(0x0e) = 0.275899379282943f * src(2) - 1.38703984532215f * src(14);
                lhs(0x0f) = -0.138617169199091f * src(1) + 1.40740373752638f * src(15);
                lhs(0x12) = lhs(0x00) + lhs(0x08);
                lhs(0x13) = lhs(0x01) + lhs(0x07);
                lhs(0x14) = lhs(0x02) + lhs(0x06);
                lhs(0x15) = lhs(0x03) + lhs(0x05);
                lhs(0x16) = 1.4142135623731f * lhs(0x04);
                lhs(0x17) = lhs(0x00) - lhs(0x08);
                lhs(0x18) = lhs(0x01) - lhs(0x07);
                lhs(0x19) = lhs(0x02) - lhs(0x06);
                lhs(0x1a) = lhs(0x03) - lhs(0x05);
                lhs(0x1d) = lhs(0x12) + lhs(0x16);
                lhs(0x1e) = lhs(0x13) + lhs(0x15);
                lhs(0x1f) = 1.4142135623731f * lhs(0x14);
                lhs(0x20) = lhs(0x12) - lhs(0x16);
                lhs(0x21) = lhs(0x13) - lhs(0x15);
                lhs(0x22) = 0.25f * (lhs(0x1d) - lhs(0x1f));
                lhs(0x23) = 0.25f * (lhs(0x20) + lhs(0x21));
                lhs(0x24) = 0.25f * (lhs(0x20) - lhs(0x21));
                lhs(0x25) = 1.4142135623731f * lhs(0x17);
                lhs(0x26) = 1.30656296487638f * lhs(0x18) + 0.541196100146197f * lhs(0x1a);
                lhs(0x27) = 1.4142135623731f * lhs(0x19);
                lhs(0x28) = -0.541196100146197f * lhs(0x18) + 1.30656296487638f * lhs(0x1a);
                lhs(0x29) = 0.176776695296637f * (lhs(0x25) + lhs(0x27)) + 0.25f * lhs(0x26);
                lhs(0x2a) = 0.25f * (lhs(0x25) - lhs(0x27));
                lhs(0x2b) = 0.176776695296637f * (lhs(0x25) + lhs(0x27)) - 0.25f * lhs(0x26);
                lhs(0x2c) = 0.353553390593274f * lhs(0x28);
                lhs(0x1b) = 0.707106781186547f * (lhs(0x2a) - lhs(0x2c));
                lhs(0x1c) = 0.707106781186547f * (lhs(0x2a) + lhs(0x2c));
                lhs(0x2d) = 1.4142135623731f * lhs(0x0c);
                lhs(0x2e) = lhs(0x0b) + lhs(0x0d);
                lhs(0x2f) = lhs(0x0a) + lhs(0x0e);
                lhs(0x30) = lhs(0x09) + lhs(0x0f);
                lhs(0x31) = lhs(0x09) - lhs(0x0f);
                lhs(0x32) = lhs(0x0a) - lhs(0x0e);
                lhs(0x33) = lhs(0x0b) - lhs(0x0d);
                lhs(0x37) = 1.4142135623731f * lhs(0x2d);
                lhs(0x38) = 1.30656296487638f * lhs(0x2e) + 0.541196100146197f * lhs(0x30);
                lhs(0x39) = 1.4142135623731f * lhs(0x2f);
                lhs(0x3a) = -0.541196100146197f * lhs(0x2e) + 1.30656296487638f * lhs(0x30);
                lhs(0x3b) = 0.176776695296637f * (lhs(0x37) + lhs(0x39)) + 0.25f * lhs(0x38);
                lhs(0x3c) = 0.25f * (lhs(0x37) - lhs(0x39));
                lhs(0x3d) = 0.176776695296637f * (lhs(0x37) + lhs(0x39)) - 0.25f * lhs(0x38);
                lhs(0x3e) = 0.353553390593274f * lhs(0x3a);
                lhs(0x34) = 0.707106781186547f * (lhs(0x3c) - lhs(0x3e));
                lhs(0x35) = 0.707106781186547f * (lhs(0x3c) + lhs(0x3e));
                lhs(0x3f) = 1.4142135623731f * lhs(0x32);
                lhs(0x40) = lhs(0x31) + lhs(0x33);
                lhs(0x41) = lhs(0x31) - lhs(0x33);
                lhs(0x42) = 0.25f * (lhs(0x3f) + lhs(0x40));
                lhs(0x43) = 0.25f * (lhs(0x3f) - lhs(0x40));
                lhs(0x44) = 0.353553390593274f * lhs(0x41);

                lhs(0) = (inverse_hann[0] * 0.176776695296637f) * (lhs(0x1d) + lhs(0x1f)) + 0.25f * lhs(0x1e);
                lhs(1) = (inverse_hann[1] * 0.707106781186547f) * (lhs(0x29) + lhs(0x3d));
                lhs(2) = (inverse_hann[2] * 0.707106781186547f) * (lhs(0x29) - lhs(0x3d));
                lhs(3) = (inverse_hann[3] * 0.707106781186547f) * (lhs(0x23) - lhs(0x43));
                lhs(4) = (inverse_hann[4] * 0.707106781186547f) * (lhs(0x23) + lhs(0x43));
                lhs(5) = (inverse_hann[5] * 0.707106781186547f) * (lhs(0x1b) - lhs(0x35));
                lhs(6) = (inverse_hann[6] * 0.707106781186547f) * (lhs(0x1b) + lhs(0x35));
                lhs(7) = (inverse_hann[7] * 0.707106781186547f) * (lhs(0x22) + lhs(0x44));
                lhs(8) = (inverse_hann[8] * 0.707106781186547f) * (lhs(0x22) - lhs(0x44));
                lhs(9) = (inverse_hann[9] * 0.707106781186547f) * (lhs(0x1c) + lhs(0x34));
                lhs(10) = (inverse_hann[10] * 0.707106781186547f) * (lhs(0x1c) - lhs(0x34));
                lhs(11) = (inverse_hann[11] * 0.707106781186547f) * (lhs(0x24) + lhs(0x42));
                lhs(12) = (inverse_hann[12] * 0.707106781186547f) * (lhs(0x24) - lhs(0x42));
                lhs(13) = (inverse_hann[13] * 0.707106781186547f) * (lhs(0x2b) - lhs(0x3b));
                lhs(14) = (inverse_hann[14] * 0.707106781186547f) * (lhs(0x2b) + lhs(0x3b));
                lhs(15) = ((inverse_hann[15] * 0.176776695296637f) * (lhs(0x1d) + lhs(0x1f)) -
                           (inverse_hann[15] * 0.25f) * lhs(0x1e));
            }
            f(xi, yi, x, y, c) = cast<float16_t>(scratch(xi, yi, x, y, c));

            scratch.compute_at(f, x);
        };

        const bool use_fast_DCT = gpu_schedule == Schedule::CUDA;

        // Perform the DCT along the rows
        RDom r(0, 16);
        Func dct_rows("dct_rows");
        if (use_fast_DCT) {
            fast_DCT(1, dct_rows, tiled, true);
        } else {
            dct_rows(xi, yi, x, y, c) += DCT(r, xi) * tiled(r, yi, x, y, c);
        }

        // Perform the DCT along the cols
        Func dct_cols("dct_cols");
        if (use_fast_DCT) {
            fast_DCT(0, dct_cols, dct_rows, true);
        } else {
            dct_cols(xi, yi, x, y, c) += DCT(r, yi) * dct_rows(xi, r, x, y, c);
        }

        // Push small coefficients to zero
        Func thresholded("thresholded");
        Expr e = dct_cols(xi, yi, x, y, c);
        Expr t = cast<float16_t>(strength * noise_floor(xi) * noise_floor(yi));
        thresholded(xi, yi, x, y, c) = select(abs(e) < t, cast<float16_t>(0.f), e);

        // Take an inverse DCT of each tile.
        Func idct_rows("idct_rows");
        if (use_fast_DCT) {
            fast_DCT(1, idct_rows, thresholded, false);
        } else {
            idct_rows(xi, yi, x, y, c) += IDCT(r, xi) * thresholded(r, yi, x, y, c);
        }

        Func idct_cols("idct_cols");
        if (use_fast_DCT) {
            fast_DCT(0, idct_cols, idct_rows, false);
        } else {
            idct_cols(xi, yi, x, y, c) += IDCT(r, yi) * idct_rows(xi, r, x, y, c);
        }

        // Add together overlapping estimates
        Func averaged("averaged");
        averaged(xi, yi, x, y, c) = (idct_cols(xi, yi, x + 1, y + 1, c) +
                                     idct_cols(xi + 8, yi, x, y + 1, c) +
                                     idct_cols(xi, yi + 8, x + 1, y, c) +
                                     idct_cols(xi + 8, yi + 8, x, y, c));

        // Un-tile and clamp
        output(x, y, c) = cast<float16_t>(clamp(averaged(x % 8, y % 8, x / 8, y / 8, c), 0.f, 1.f));

        // Make it so that we can unroll across c
        output.dim(2).set_bounds(0, 3);
        // Make the final stores vectorizable
        output.dim(2).set_stride(output.dim(2).stride() / 4 * 4);
        output.dim(1).set_stride(output.dim(1).stride() / 4 * 4);
        output.set_host_alignment(64);
        output.dim(0).set_min(0);
        output.dim(1).set_min(0);
        // Make the IR easier to read
        input.dim(0).set_min(0);
        input.dim(1).set_min(0);
        input.dim(2).set_bounds(0, 3);

        Var xii{"xii"}, yii{"yii"}, z{"z"}, xo{"xo"}, yo{"yo"};

        // The schedule. We should be able to fuse this whole thing into two
        // kernel launches - one to compute the per-tile work, and a final one
        // to do the overlap-and-add. We'll schedule the overlap-and-add the
        // same way for both schedules:
        output.compute_root()
            .tile(x, y, xi, yi, 16, 16, TailStrategy::RoundUp)
            .reorder(c, xi, yi, x, y)
            .gpu_blocks(x, y)
            .vectorize(xi, 8)
            .unroll(c)
            .fuse(xi, yi, z)
            .gpu_threads(z);

        switch (gpu_schedule) {
        case Schedule::CUDA: {
            if (use_fast_DCT) {
                // The fast DCT algorithm is serial along the dimension we're
                // transforming, so all parallelism goes to the other dimension.

                auto schedule_fast_transform = [&](Func w, Func t, int tx, int ty) {
                    w
                        .tile(xi, yi, xii, yii, tx, ty)
                        .reorder(xii, yii, y, xi, yi, x, c)
                        .unroll(y)
                        .vectorize(xii)
                        .vectorize(yii)
                        .fuse(xi, yi, z)
                        .fuse(z, x, z)
                        .gpu_threads(z, c);
                    t.compute_at(w, z)
                        .unroll(c)
                        .unroll(xi)
                        .unroll(yi)
                        .unroll(x)
                        .unroll(y);
                };

                idct_cols
                    .in()
                    .compute_root()
                    .tile(x, y, xo, yo, x, y, 2, 1)
                    .reorder(xi, yi, x, y, c, xo, yo)
                    .gpu_blocks(xo, yo);

                for (Func t : {idct_rows.in(), dct_rows.in(), thresholded}) {
                    t.compute_at(idct_cols.in(), xo);
                }

                schedule_fast_transform(idct_rows.in(), idct_rows, 1, 16);
                schedule_fast_transform(idct_cols.in(), idct_cols, 16, 1);
                schedule_fast_transform(thresholded, dct_cols, 16, 1);
                schedule_fast_transform(dct_rows.in(), dct_rows, 1, 16);
            } else {
                // For the brute-force transforms, we want each thread handling
                // a small tile of outputs. To amortize the loads of the
                // transform matrices we'll unroll in c and a little in x and
                // y. To amortize the loads of the image tile we'll unroll in
                // either xi or yi, depending on which direction we're
                // transforming in.

                auto schedule_transform = [&](Func w, Func t, int tx, int ty) {
                    w
                        .tile(xi, yi, xii, yii, tx, ty)
                        .reorder(c, xii, yii, x, y, xi, yi)
                        .unroll(c)
                        .unroll(x)
                        .unroll(y)
                        .vectorize(xii)
                        .vectorize(yii)
                        .fuse(xi, yi, z)
                        .gpu_threads(z);
                    t.compute_at(w, z)
                        .unroll(c)
                        .unroll(xi)
                        .unroll(yi)
                        .unroll(x)
                        .unroll(y)
                        .update()
                        .reorder(c, x, y, xi, yi, r)
                        .unroll(c)
                        .unroll(xi)
                        .unroll(yi)
                        .unroll(x)
                        .unroll(y);
                };

                idct_cols
                    .in()
                    .compute_root()
                    .tile(x, y, xo, yo, x, y, 2, 1)
                    .reorder(xi, yi, x, y, c, xo, yo)
                    .gpu_blocks(xo, yo);

                for (Func t : {idct_rows.in(), dct_rows.in(), thresholded}) {
                    t.compute_at(idct_cols.in(), xo);
                }

                schedule_transform(idct_rows.in(), idct_rows, 8, 1);
                schedule_transform(idct_cols.in(), idct_cols, 1, 8);
                schedule_transform(thresholded, dct_cols, 1, 8);
                schedule_transform(dct_rows.in(), dct_rows, 8, 1);
            }

            break;
        }
        case Schedule::TensorCore: {
            LoopLevel blocks;

            auto schedule_transform = [&](Func t) {
                t.in()
                    .compute_at(blocks)
                    .unroll(c)
                    .unroll(x)
                    .unroll(y)
                    .vectorize(xi)
                    .vectorize(yi);
                t.compute_at(blocks)
                    .store_in(MemoryType::WMMAAccumulator)
                    .unroll(c)
                    .unroll(x)
                    .unroll(y)
                    .vectorize(xi)
                    .vectorize(yi)
                    .update()
                    .reorder(r, xi, yi, c, x, y)
                    .unroll(c)
                    .unroll(x)
                    .unroll(y)
                    .vectorize(xi)
                    .vectorize(yi)
                    .atomic()
                    .vectorize(r);
            };

            idct_cols
                .in()
                .compute_root()
                .tile(x, y, xo, yo, x, y, 2, 1)
                .reorder(xi, yi, x, y, c, xo, yo)
                .gpu_blocks(xo, yo);

            blocks.set({idct_cols.in(), xo});

            schedule_transform(idct_rows);
            schedule_transform(idct_cols);
            schedule_transform(dct_cols);
            schedule_transform(dct_rows);

            idct_cols.in().compute_root();

            thresholded
                .compute_at(blocks)
                .split(xi, xi, xii, 8)
                .fuse(xi, yi, z)
                .vectorize(xii)
                .gpu_lanes(z)
                .unroll(c)
                .unroll(x)
                .unroll(y);
        }
        }
    }
};

HALIDE_REGISTER_GENERATOR(Denoise, denoise);
