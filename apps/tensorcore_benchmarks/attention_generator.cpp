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
    GeneratorParam<int> L{"L", 64};    // Sequence length
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
        rj = RDom(0, L);

        // Scale factor 1/sqrt(D)
        Expr scale = 1.0f / Halide::sqrt(cast<float>(D));

        // scores(t, j, n) = (Q(:,t,n) · K(:,j,n)) / sqrt(D)
        scores(t, j, n) = 0.0f;
        scores(t, j, n) += cast<float>(query(rd, t, n)) * cast<float>(key(rd, j, n));
        
        //scaled_scores(t, j, n) = scores(t, j, n) * scale;

        // Row-wise max for numerical stability: max over j
        //row_max(t, n) = -std::numeric_limits<float>::infinity();
        //row_max(t, n) = max(row_max(t, n), scaled_scores(t, rj, n));

        // Exponentiated, shifted scores and row-wise sum
        //exp_scores(t, j, n) = exp(scaled_scores(t, j, n) - row_max(t, n));
        //row_sum(t, n) = 0.0f;
        //row_sum(t, n) += exp_scores(t, rj, n);

        // Probabilities via softmax
        //prob(t, j, n) = exp_scores(t, j, n) / row_sum(t, n);

        // Output: weighted sum of values along j
        //output(d, t, n) = 0.0f;
        //output(d, t, n) += prob(t, rj, n) * cast<float>(value(d, rj, n));
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
    }

private:
    // Intermediate funcs
    Var d{"d"}, t{"t"}, n{"n"}, j{"j"};

    RDom rd{"rd"}, rj{"rj"};

    Func scores{"scores"};
    Func scaled_scores{"scaled_scores"};
    Func row_max{"row_max"};
    Func exp_scores{"exp_scores"};
    Func row_sum{"row_sum"};
    Func prob{"prob"};
};

}  // namespace

HALIDE_REGISTER_GENERATOR(Attention, attention)