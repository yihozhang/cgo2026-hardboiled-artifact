#include "Halide.h"

#include "common.h"
#include <limits>

namespace {

using namespace Halide;

class Attention : public Halide::Generator<Attention> {
public:
    // Generator Params
    GeneratorParam<Schedule> gpu_schedule{"gpu_schedule", Schedule::CUDA, {
        {"cuda_only", Schedule::CUDA},
        {"tensorcore", Schedule::TensorCore}
    }};
    
    // Model dimensions
    GeneratorParam<int> D{"D", 128};   // Feature dimension
    GeneratorParam<int> L{"L", 2048};  // Sequence length
    GeneratorParam<int> N{"N", 8};     // Batch size

    // Inputs: Q, K, V shaped [D, L, N]
    Input<Buffer<float16_t, 3>> query{"query"};
    Input<Buffer<float16_t, 3>> key{"key"};
    Input<Buffer<float16_t, 3>> value{"value"};

    // Output: [D, L, N]
    Output<Buffer<float, 3>> output{"output"};

    void generate() {
        // Reductions over feature and time positions
        rd = RDom(0, D);
        rk = RDom(0, L);

        // Scale factor 1/sqrt(D)
        Expr scale = 1.0f / Halide::sqrt(cast<float>(D));

        // scores(k, q, n) = (Q(:,q,n) · K(:,k,n)) / sqrt(D)
        scores(k, q, n) = 0.0f;
        scores(k, q, n) += cast<float>(query(rd, q, n)) * cast<float>(key(rd, k, n));
        
        scaled_scores(k, q, n) = scores(k, q, n) * scale;

        // Row-wise max for numerical stability: max over k
        row_max(q, n) = -std::numeric_limits<float>::infinity();
        row_max(q, n) = max(row_max(q, n), scaled_scores(rk, q, n));

        // Exponentiated, shifted scores
        exp_scores(k, q, n) = exp(scaled_scores(k, q, n) - row_max(q, n));
        
        // Row-wise sum
        row_sum(q, n) = 0.0f;
        row_sum(q, n) += exp_scores(rk, q, n);

        // Probabilities via softmax
        prob(k, q, n) = cast<float16_t>(exp_scores(k, q, n) / row_sum(q, n));

        // Output: weighted sum of values along rk
        weighted_sum(d, l, n) = 0.0f;
        weighted_sum(d, l, n) += cast<float>(prob(rk, l, n)) * cast<float>(value(d, rk, n));

        output(d, l, n) = weighted_sum(d, l, n);
    }

    void schedule() {
        // Set simple bounds/strides; no GPU scheduling here
        const int _D = D, _L = L, _N = N;

        output.dim(0).set_bounds(0, _D).set_stride(1);
        output.dim(1).set_bounds(0, _L).set_stride(_D);
        output.dim(2).set_bounds(0, _N).set_stride(_D * _L);

        query.dim(0).set_bounds(0, _D).set_stride(1);
        query.dim(1).set_bounds(0, _L).set_stride(_D);
        query.dim(2).set_bounds(0, _N).set_stride(_D * _L);

        key.dim(0).set_bounds(0, _D).set_stride(1);
        key.dim(1).set_bounds(0, _L).set_stride(_D);
        key.dim(2).set_bounds(0, _N).set_stride(_D * _L);

        value.dim(0).set_bounds(0, _D).set_stride(1);
        value.dim(1).set_bounds(0, _L).set_stride(_D);
        value.dim(2).set_bounds(0, _N).set_stride(_D * _L);

        Var di, li, ki, qi;
        Var mmdi, mmli, mmki, mmqi;
        RVar rri, rro, rroo;
        int mm_tile = 16;

        output.split(d, d, di, mm_tile * 1)
            .split(di, di, mmdi, mm_tile)
            .split(l, l, li, mm_tile * 1)
            .split(li, li, mmli, mm_tile)
            .gpu_blocks(d, l, n)
            .reorder({mmdi, mmli, di, li, d, l, n})
            .unroll(di)
            .unroll(li)
            .vectorize(mmdi)
            .vectorize(mmli);

        weighted_sum.compute_at(output, d)
            .store_in(MemoryType::WMMAAccumulator)
            .split(l, l, li, mm_tile)
            .split(d, d, di, mm_tile)
            .vectorize(di)
            .vectorize(li)
            .unroll(d)
            .unroll(l);

        weighted_sum.update()
            .split(l, l, mmli, mm_tile)
            .split(d, d, mmdi, mm_tile)
            .split(rk, rro, rri, mm_tile)
            .reorder({rri, mmdi, mmli, rro, d, l})
            .unroll(rro)
            .unroll(l)
            .unroll(d)
            .atomic()
            .vectorize(rri)
            .vectorize(mmli)
            .vectorize(mmdi);

        row_max.compute_at(weighted_sum, d);
        row_sum.compute_at(weighted_sum, d);
        prob.compute_at(weighted_sum, d);

        scaled_scores.compute_root()
            .split(q, q, qi, mm_tile * 1)
            .split(qi, qi, mmqi, mm_tile)
            .split(k, k, ki, mm_tile * 1)
            .split(ki, ki, mmki, mm_tile)
            .gpu_blocks(k, q, n)
            .reorder({mmki, mmqi, ki, qi, k, q, n})
            .unroll(ki)
            .unroll(qi)
            .vectorize(mmki)
            .vectorize(mmqi)
            ;

        // initialization
        scores.compute_at(scaled_scores, k)
            .store_in(MemoryType::WMMAAccumulator)
            .split(q, q, qi, mm_tile)
            .split(k, k, ki, mm_tile)
            .vectorize(qi)
            .vectorize(ki)
            .unroll(q)
            .unroll(k);

        scores.update()
            .split(q, q, mmqi, mm_tile)
            .split(k, k, mmki, mm_tile)
            .split(rd, rro, rri, mm_tile)
            .reorder({rri, mmki, mmqi, rro, k, q})
            .unroll(rro)
            .unroll(q)
            .unroll(k)
            .atomic()
            .vectorize(rri)
            .vectorize(mmqi)
            .vectorize(mmki);
    }

private:
    // Intermediate funcs
    Var d{"d"}, l{"l"}, n{"n"}, q{"q"}, k{"k"};

    RDom rd, rk;

    Func scores{"scores"};
    Func scaled_scores{"scaled_scores"};
    Func row_max{"row_max"};
    Func exp_scores{"exp_scores"};
    Func row_sum{"row_sum"};
    Func prob{"prob"};
    Func weighted_sum{"weighted_sum"};
};

}  // namespace

HALIDE_REGISTER_GENERATOR(Attention, attention)