#include "Halide.h"
#include <stdio.h>
#include <stdlib.h> // For rand()

using namespace Halide;

int main(int argc, char **argv) {
    // The size of the N x N matrices
    const int N = 256;

    // 1. Create and fill host-side buffers with random data
    Buffer<float> A_buf(N, N);
    Buffer<float> B_buf(N, N);
    Buffer<float> C_buf(N, N);
    Buffer<float> D_buf(N, N);
    Buffer<float> E_buf(N, N);
    Buffer<float> F_buf(N, N);
    Buffer<float> G_buf(N, N);
    Buffer<float> H_buf(N, N);
    Buffer<float> I_buf(N, N);

    for (int y = 0; y < N; y++) { // row
        for (int x = 0; x < N; x++) { // col
            A_buf(x, y) = (rand() % 100) / 100.0f - 0.5f;
            B_buf(x, y) = (rand() % 100) / 100.0f - 0.5f;
            C_buf(x, y) = (rand() % 100) / 100.0f - 0.5f;
            D_buf(x, y) = (rand() % 100) / 100.0f - 0.5f;
            E_buf(x, y) = (rand() % 100) / 100.0f - 0.5f;
            F_buf(x, y) = (rand() % 100) / 100.0f - 0.5f;
            G_buf(x, y) = (rand() % 100) / 100.0f - 0.5f;
            H_buf(x, y) = (rand() % 100) / 100.0f - 0.5f;
            I_buf(x, y) = (rand() % 100) / 100.0f - 0.5f;
        }
    }

    // 2. Define ImageParams (inputs) for the Halide pipeline
    ImageParam A(type_of<float>(), 2);
    ImageParam B(type_of<float>(), 2);
    ImageParam C(type_of<float>(), 2);
    ImageParam D(type_of<float>(), 2);
    ImageParam E(type_of<float>(), 2);
    ImageParam F(type_of<float>(), 2);
    ImageParam G(type_of<float>(), 2);
    ImageParam H(type_of<float>(), 2);
    ImageParam I(type_of<float>(), 2);

    Var i("i"), j("j"), k("k");
    RDom l(0, N, "l");
    Func def("def");
    def(i, j, k) = 0.0f;
    def(i, j, k) += D(l, i) * E(l, j) * F(l, k);

    RDom m(0, N, "m");
    Func ghi("ghi");
    ghi(i, j, k) = 0.0f;
    ghi(i, j, k) += G(m, i) * H(m, j) * I(m, k);

    Func abc("abc");
    abc(i, j, k) = A(j, i) * B(k, i) * C(k, j);

    RDom r(0, N, 0, N, 0, N, "r");
    Func res("final");
    res() = 0.0f;
    res() += abc(r.x, r.y, r.z) * def(r.x, r.y, r.z) * ghi(r.x, r.y, r.z);

    // res.compile_to_lowered_stmt("example_stmt.html", {A, B, C, D, E, F, G, H, I}, HTML);

    // 4. Schedule the pipeline
    Var io("io"), ii("ii"), jo("jo"), ji("ji"), ko("ko"), ki("ki");
    def.compute_root()
       .gpu_tile(i, j, k, io, jo, ko, ii, ji, ki, 8, 8, 8)
       .update(0)
       .gpu_tile(i, j, k, io, jo, ko, ii, ji, ki, 8, 8, 8)
       .reorder(l, ii, ji, ki, io, jo, ko);
    ghi.compute_root()
       .gpu_tile(i, j, k, io, jo, ko, ii, ji, ki, 8, 8, 8)
       .update(0)
       .gpu_tile(i, j, k, io, jo, ko, ii, ji, ki, 8, 8, 8)
       .reorder(m, ii, ji, ki, io, jo, ko);;
    res.compute_root();

    // 5. Set the input buffers
    A.set(A_buf);
    B.set(B_buf);
    C.set(C_buf);
    D.set(D_buf);
    E.set(E_buf);
    F.set(F_buf);
    G.set(G_buf);
    H.set(H_buf);
    I.set(I_buf);

    // // 6. Realize the pipeline
    printf("Running Halide pipeline...\n");
    // Buffer<float> output = res.realize({}, get_jit_target_from_environment().with_feature(Target::Feature::CUDA));
    Buffer<float> output = res.realize();
    printf("Halide pipeline finished.\n");

    // The scalar result is at index 0
    float halide_result = output(0);
    printf("Halide result: %f\n", halide_result);

    return 0;
}
