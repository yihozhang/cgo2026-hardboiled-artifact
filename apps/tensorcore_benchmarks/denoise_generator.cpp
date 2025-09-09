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

        // Take a DCT of each tile. We'll use a direct DCT, computing the
        // matrices at compile-time. Decomposing this into a fast DCT would
        // involve lots of expensive shuffling of data and synchronization
        // points between threads. Multiplies are cheap. Moving data is costly.
        // Precomputing the entire matrix also lets us bake a Hann window in. If
        // we window each row of the forwards matrix, that's the same as
        // windowing each column of the signal.
        Buffer<float16_t> IDCT_mat(16, 16), DCT_mat(16, 16);
        float hann[16], inverse_hann[16];
        for (int i = 0; i < 16; i++) {
            hann[i] = 0.5 * (1 - cos(2 * M_PI * (i + 1) / 17));
        }
        // Because of the overlap-add, the inverse window is a little more complicated.
        // Each point is going to be the sum of four weighted values, where the sum of
        // the weights for pixel i, j is:
        // (hann[i] + hann[(i + 8) & 16]) * (hann[j] + hann[(j + 8) % 16])

        // To correct for this, we want to multiply each pixel in the tile by
        // the inverse of the above. Because it's separable, we can treat this
        // as multiplying the rows by one thing, and the columns by another.
        // Let H be the diagonal matrix H(i, i) = 1 / (hann[i] + hann[(i+8)%16])
        // Multiplying by H corrects the Hann window down each col.  Let D be
        // the IDCT matrix, which if you view a tile as a small matrix, takes a
        // DCT down the columns. Therefore, to get a signal M out of the
        // transform domain, we want to do this:
        //   (H (H (D (D M)')')')'
        // = (H D) ((H D) M')'
        // I.e. we just need to post-multiply D by H, and we should be good to
        // go. We can bake this into D.

        for (int i = 0; i < 16; i++) {
            inverse_hann[i] = 1 / (hann[i] + hann[(i + 8) % 16]);
        }

        for (int j = 0; j < 16; j++) {  // Row
            float alpha = (j == 0) ? (1.0f / sqrt(16.0f)) : sqrt(2.0f / 16.0f);
            for (int i = 0; i < 16; i++) {  // Col
                float f = alpha * cos((M_PI / 16) * (i + 0.5f) * j);
                DCT_mat(i, j) = float16_t(f * hann[i]);
                IDCT_mat(j, i) = float16_t(f * inverse_hann[i]);
            }
        }

        // Now we want to know how much energy to expect in each bin after
        // taking the transform, so that we can set thresholds. Consider
        // Gaussian random noise in [-1, 1]. After taking a DCT the expected
        // distribution per bin is still a Gaussian, with a variance equal to
        // the sum of the squares of that row of the matrix. If it were the pure
        // DCT matrix that'd be one, but we messed it up with the windowing.
        Buffer<float> noise_floor(16);
        for (int j = 0; j < 16; j++) {
            float n = 0;
            for (int i = 0; i < 16; i++) {
                float v = float(DCT_mat(i, j));
                n += v * v;
            }
            noise_floor(j) = std::sqrt(n);
        }

        Func DCT("DCT");
        DCT(x, y) = DCT_mat(x, y);

        Func IDCT("IDCT");
        IDCT(x, y) = IDCT_mat(x, y);

        // Perform the DCT along the rows
        RDom r(0, 16);

        Func dct_rows("dct_rows");
        dct_rows(xi, yi, x, y, c) += DCT(r, xi) * tiled(r, yi, x, y, c);

        // Perform the DCT along the cols
        Func dct_cols("dct_cols");
        dct_cols(xi, yi, x, y, c) += DCT(r, yi) * dct_rows(xi, r, x, y, c);

        // Push small coefficients to zero
        Func thresholded("thresholded");
        Expr e = dct_cols(xi, yi, x, y, c);
        Expr t = strength * noise_floor(xi) * noise_floor(yi);
        thresholded(xi, yi, x, y, c) = select(abs(e) < t, cast<float16_t>(0.f), e);

        // Take an inverse DCT of each tile.
        Func idct_rows("idct_rows");
        idct_rows(xi, yi, x, y, c) += IDCT(r, xi) * thresholded(r, yi, x, y, c);

        Func idct_cols("idct_cols");
        idct_cols(xi, yi, x, y, c) += IDCT(r, yi) * idct_rows(xi, r, x, y, c);

        // Add together overlapping estimates
        Func averaged("averaged");
        averaged(xi, yi, x, y, c) = (idct_cols(xi, yi, x + 1, y + 1, c) +
                                     idct_cols(xi + 8, yi, x, y + 1, c) +
                                     idct_cols(xi, yi + 8, x + 1, y, c) +
                                     idct_cols(xi + 8, yi + 8, x, y, c));

        // Un-tile and clamp
        output(x, y, c) = cast<float16_t>(clamp(averaged(x % 8, y % 8, x / 8, y / 8, c), 0.f, 1.f));

        // Debugging schedule
        /*
        for (Func f : {tiled, dct_rows, dct_cols, thresholded, idct_rows, idct_cols, averaged}) {
            f.compute_root();
        }
        */

        // Make it so that we can unroll across c
        output.dim(2).set_bounds(0, 3);
        // Make the IR easier to read
        output.dim(0).set_min(0);
        output.dim(1).set_min(0);
        input.dim(0).set_min(0);
        input.dim(1).set_min(0);
        input.dim(2).set_bounds(0, 3);

        // The schedule. We should be able to fuse this whole thing into two
        // kernel launches - one to compute the per-tile work, and a final one
        // to do the overlap-and-add. We'll schedule the overlap-and-add the
        // same way for both schedules:
        output.compute_root()
            .tile(x, y, xi, yi, 16, 16, TailStrategy::RoundUp)
            .gpu_blocks(x, y, c)
            .gpu_threads(xi, yi);

        Var xii{"xii"}, yii{"yii"}, z{"z"}, xo{"xo"}, yo{"yo"};

        switch (gpu_schedule) {
        case Schedule::CUDA: {
            // For each transform, we want each thread handling a small tile of
            // outputs. To amortize the loads of the transform matrices we'll
            // unroll in c and a little in x and y. To amortize the loads of the
            // image tile we'll unroll in either xi or yi, depending on which
            // direction we're transforming in.

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

            // For the row transforms, we want the transform matrix transposed
            // (TODO: check this is really necessary)
            DCT.in(dct_rows).compute_root().reorder_storage(y, x);
            IDCT.in(idct_rows).compute_root().reorder_storage(y, x);

            schedule_transform(idct_rows);
            schedule_transform(idct_cols);
            schedule_transform(dct_cols);
            schedule_transform(dct_rows);

            idct_cols.in().compute_root();

            thresholded
                .compute_at(blocks)
                .split(yi, yi, yii, 2)
                .reorder(c, y, x, yi, yii, xi)
                .fuse(xi, yii, z)
                .gpu_lanes(z)
                .unroll(c)
                .unroll(x)
                .unroll(y)
                .unroll(yi);
        }
        }
    }
};

HALIDE_REGISTER_GENERATOR(Denoise, denoise);
