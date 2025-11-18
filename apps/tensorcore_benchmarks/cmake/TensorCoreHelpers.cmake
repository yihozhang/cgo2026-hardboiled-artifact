# Helper functions for building Halide libraries with TensorCore support

function(add_conv_library LIB_NAME GENERATOR_NAME TARGET_PLATFORM)
    add_halide_library(${LIB_NAME}
        FROM ${GENERATOR_NAME}.generator
        GENERATOR ${GENERATOR_NAME}
        FUNCTION_NAME ${GENERATOR_NAME}
        TARGETS ${TARGET_PLATFORM}
        PARAMS 
            kSize=${CONV_KERNEL_SIZE} 
            gpu_schedule=${SCHEDULE}
        EXTRA_OUTPUTS assembly llvm_assembly stmt
    )
endfunction()

function(add_denoise_library LIB_NAME GENERATOR_NAME TARGET_PLATFORM)
    add_halide_library(${LIB_NAME}
        FROM ${GENERATOR_NAME}.generator
        GENERATOR ${GENERATOR_NAME}
        FUNCTION_NAME ${GENERATOR_NAME}
        TARGETS ${TARGET_PLATFORM}
        PARAMS 
            gpu_schedule=${SCHEDULE}
        EXTRA_OUTPUTS assembly llvm_assembly stmt
    )
endfunction()

function(add_resize_library LIB_NAME GENERATOR_NAME TARGET_PLATFORM)
    add_halide_library(${LIB_NAME}
        FROM ${GENERATOR_NAME}.generator
        GENERATOR ${GENERATOR_NAME}
        FUNCTION_NAME ${GENERATOR_NAME}
        TARGETS ${TARGET_PLATFORM}
        PARAMS 
            upsample=${UPSAMPLE}
            gpu_schedule=${SCHEDULE}
        EXTRA_OUTPUTS assembly llvm_assembly stmt
    )
endfunction()

function(add_matmul_library LIB_NAME GENERATOR_NAME TARGET_PLATFORM)
    add_halide_library(${LIB_NAME}
        FROM ${GENERATOR_NAME}.generator
        GENERATOR ${GENERATOR_NAME}
        FUNCTION_NAME ${GENERATOR_NAME}
        TARGETS ${TARGET_PLATFORM}
        PARAMS
            M=${MATMUL_M}
            N=${MATMUL_N}
            K=${MATMUL_K}
            gpu_schedule=${SCHEDULE}
        EXTRA_OUTPUTS assembly llvm_assembly stmt
    )
endfunction()


function(add_nnl_library LIB_NAME GENERATOR_NAME TARGET_PLATFORM)
    add_halide_library(${LIB_NAME}
        FROM ${GENERATOR_NAME}.generator
        GENERATOR ${GENERATOR_NAME}
        FUNCTION_NAME ${GENERATOR_NAME}
        TARGETS ${TARGET_PLATFORM}
        PARAMS 
            N=${NN_TENSOR_N}
            H=${NN_TENSOR_H}
            W=${NN_TENSOR_W}
            C=${NN_TENSOR_C}
            gpu_schedule=${SCHEDULE}
        EXTRA_OUTPUTS assembly llvm_assembly stmt
    )
endfunction() 

function(add_attn_library LIB_NAME GENERATOR_NAME TARGET_PLATFORM)
    add_halide_library(${LIB_NAME}
        FROM ${GENERATOR_NAME}.generator
        GENERATOR ${GENERATOR_NAME}
        FUNCTION_NAME ${GENERATOR_NAME}
        TARGETS ${TARGET_PLATFORM}
        PARAMS 
            D=${ATT_D}
            L=${ATT_L}
            N=${ATT_N}
            gpu_schedule=${SCHEDULE}
        EXTRA_OUTPUTS assembly llvm_assembly stmt
    )
endfunction()