# Helper functions for building Halide libraries with TensorCore support

function(add_conv_library LIB_NAME GENERATOR_NAME TARGET_PLATFORM)
    add_halide_library(${LIB_NAME}
        FROM ${GENERATOR_NAME}.generator
        GENERATOR ${GENERATOR_NAME}
        FUNCTION_NAME ${GENERATOR_NAME}
        TARGETS ${TARGET_PLATFORM}
        PARAMS 
            kSize=${CONV_KERNEL_SIZE} 
            imgRow=${CONV_IMG_ROW} 
            imgCol=${CONV_IMG_COL} 
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