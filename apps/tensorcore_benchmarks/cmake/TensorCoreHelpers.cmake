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