tests(GROUPS instrsel-benchmarks
        SOURCES
        einsum_flat_1x1.cpp
        matmul_flat_1x1.cpp
        matmul_flat_1x4.cpp
        matmul_flat_1x4_preload_rhs.cpp
        matmul_vnni_1x4_preload_rhs.cpp
        matmul_vnni_1x4.cpp
        matmul_vnni_1x1.cpp
        matmul_vnni_2x2.cpp
        convolution.cpp
)

# Disable PCH for this target
set_target_properties(instrsel-benchmarks_convolution PROPERTIES DISABLE_PRECOMPILE_HEADERS ON)