HL_TARGET ?= host-cuda

include ../support/Makefile.inc

KERNEL_SIZE ?= 16

all: all_cuda_only all_tensorcore
all_cuda_only: $(BIN)/cuda_only/matmul $(BIN)/cuda_only/conv1d $(BIN)/cuda_only/conv2d $(BIN)/cuda_only/downsample $(BIN)/cuda_only/upsample
all_tensorcore: $(BIN)/tensorcore/matmul $(BIN)/tensorcore/conv1d $(BIN)/tensorcore/conv2d $(BIN)/tensorcore/downsample $(BIN)/tensorcore/upsample
clean:
	rm -rf $(BIN)

$(GENERATOR_BIN)/conv1d.generator: conv1d_generator.cpp $(GENERATOR_DEPS)
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) $(filter %.cpp,$^) -o $@ $(LIBHALIDE_LDFLAGS)

.PRECIOUS: $(BIN)/%/conv1d.a
$(BIN)/%/conv1d.a: $(GENERATOR_BIN)/conv1d.generator
	@mkdir -p $(@D)
	$^ -g conv1d -e $(GENERATOR_OUTPUTS) -o $(@D) -f conv1d target=$(HL_TARGET) gpu_schedule=$* kSize=$(KERNEL_SIZE)

$(BIN)/%/conv1d: process.cpp $(BIN)/%/conv1d.a 
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) -I$(BIN)/$* -Wall $^ -o $@ $(LDFLAGS) $(IMAGE_IO_FLAGS) $(CUDA_LDFLAGS) $(OPENCL_LDFLAGS) -DBENCHMARK_HEADER="\"$(BIN)/$*/conv1d.h\"" -DBENCHMARK_NAME="\"conv1d\"" -DKERNEL_SIZE=$(KERNEL_SIZE) -DIMG_ROW=4096 -DIMG_COL=4096 -DSCHEDULE="\"$*\"" -DVERIFY_OUTPUT=false -DRUN_conv1d


$(GENERATOR_BIN)/conv2d.generator: conv2d_generator.cpp $(GENERATOR_DEPS)
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) $(filter %.cpp,$^) -o $@ $(LIBHALIDE_LDFLAGS)

.PRECIOUS: $(BIN)/%/conv2d.a
$(BIN)/%/conv2d.a: $(GENERATOR_BIN)/conv2d.generator
	@mkdir -p $(@D)
	$^ -g conv2d -e $(GENERATOR_OUTPUTS),conceptual_stmt_html -o $(@D) -f conv2d target=$(HL_TARGET) gpu_schedule=$* kSize=$(KERNEL_SIZE)

$(BIN)/%/conv2d: process.cpp $(BIN)/%/conv2d.a 
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) -I$(BIN)/$* -Wall $^ -o $@ $(LDFLAGS) $(IMAGE_IO_FLAGS) $(CUDA_LDFLAGS) $(OPENCL_LDFLAGS) -DBENCHMARK_HEADER="\"$(BIN)/$*/conv2d.h\"" -DBENCHMARK_NAME="\"conv2d\"" -DKERNEL_SIZE=$(KERNEL_SIZE) -DIMG_ROW=4096 -DIMG_COL=4096 -DSCHEDULE="\"$*\"" -DVERIFY_OUTPUT=true -DRUN_conv2d


$(GENERATOR_BIN)/upsample.generator: upsample_generator.cpp $(GENERATOR_DEPS)
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) $(filter %.cpp,$^) -o $@ $(LIBHALIDE_LDFLAGS)

.PRECIOUS: $(BIN)/%/upsample.a
$(BIN)/%/upsample.a: $(GENERATOR_BIN)/upsample.generator
	@mkdir -p $(@D)
	$^ -g upsample -e $(GENERATOR_OUTPUTS),conceptual_stmt_html -o $(@D) -f upsample target=$(HL_TARGET) gpu_schedule=$* kSize=$(KERNEL_SIZE)

$(BIN)/%/upsample: process.cpp $(BIN)/%/upsample.a 
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) -I$(BIN)/$* -Wall $^ -o $@ $(LDFLAGS) $(IMAGE_IO_FLAGS) $(CUDA_LDFLAGS) $(OPENCL_LDFLAGS) -DBENCHMARK_HEADER="\"$(BIN)/$*/upsample.h\"" -DBENCHMARK_NAME="\"upsample\"" -DKERNEL_SIZE=$(KERNEL_SIZE) -DIMG_ROW=4096 -DIMG_COL=4096 -DSCHEDULE="\"$*\"" -DVERIFY_OUTPUT=true -DRUN_upsample


$(GENERATOR_BIN)/downsample.generator: downsample_generator.cpp $(GENERATOR_DEPS)
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) $(filter %.cpp,$^) -o $@ $(LIBHALIDE_LDFLAGS)

.PRECIOUS: $(BIN)/%/downsample.a
$(BIN)/%/downsample.a: $(GENERATOR_BIN)/downsample.generator
	@mkdir -p $(@D)
	$^ -g downsample -e $(GENERATOR_OUTPUTS),conceptual_stmt_html -o $(@D) -f downsample target=$(HL_TARGET) gpu_schedule=$* kSize=$(KERNEL_SIZE)

$(BIN)/%/downsample: process.cpp $(BIN)/%/downsample.a 
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) -I$(BIN)/$* -Wall $^ -o $@ $(LDFLAGS) $(IMAGE_IO_FLAGS) $(CUDA_LDFLAGS) $(OPENCL_LDFLAGS) -DBENCHMARK_HEADER="\"$(BIN)/$*/downsample.h\"" -DBENCHMARK_NAME="\"downsample\"" -DKERNEL_SIZE=$(KERNEL_SIZE) -DIMG_ROW=4096 -DIMG_COL=4096 -DSCHEDULE="\"$*\"" -DVERIFY_OUTPUT=true -DRUN_downsample

$(GENERATOR_BIN)/resize.generator: resize_generator.cpp $(GENERATOR_DEPS)
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) $(filter %.cpp,$^) -o $@ $(LIBHALIDE_LDFLAGS)

.PRECIOUS: $(BIN)/%/resize.a
$(BIN)/%/resize.a: $(GENERATOR_BIN)/resize.generator
	@mkdir -p $(@D)
	$^ -g resize -e $(GENERATOR_OUTPUTS),conceptual_stmt_html -o $(@D) -f resize target=$(HL_TARGET) gpu_schedule=$*

$(BIN)/%/resize: process.cpp $(BIN)/%/resize.a 
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) -I$(BIN)/$* -Wall $^ -o $@ $(LDFLAGS) $(IMAGE_IO_FLAGS) $(CUDA_LDFLAGS) $(OPENCL_LDFLAGS) -DBENCHMARK_HEADER="\"$(BIN)/$*/resize.h\"" -DBENCHMARK_NAME="\"resize\"" -DKERNEL_SIZE=$(KERNEL_SIZE) -DIMG_ROW=4096 -DIMG_COL=4096 -DSCHEDULE="\"$*\"" -DVERIFY_OUTPUT=true -DRUN_resize

$(GENERATOR_BIN)/rec_filter.generator: rec_filter_generator.cpp $(GENERATOR_DEPS)
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) $(filter %.cpp,$^) -o $@ $(LIBHALIDE_LDFLAGS)

.PRECIOUS: $(BIN)/%/rec_filter.a
$(BIN)/%/rec_filter.a: $(GENERATOR_BIN)/rec_filter.generator
	@mkdir -p $(@D)
	$^ -g rec_filter -e $(GENERATOR_OUTPUTS),conceptual_stmt_html -o $(@D) -f rec_filter target=$(HL_TARGET) gpu_schedule=$* 

$(BIN)/%/rec_filter: process.cpp $(BIN)/%/rec_filter.a 
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) -I$(BIN)/$* -Wall $^ -o $@ $(LDFLAGS) $(IMAGE_IO_FLAGS) $(CUDA_LDFLAGS) $(OPENCL_LDFLAGS) -DBENCHMARK_HEADER="\"$(BIN)/$*/rec_filter.h\"" -DBENCHMARK_NAME="\"rec_filter\"" -DIMG_ROW=4096 -DIMG_COL=4096 -DSCHEDULE="\"$*\"" -DVERIFY_OUTPUT=true -DRUN_rec_filter

$(GENERATOR_BIN)/matmul.generator: matmul_generator.cpp $(GENERATOR_DEPS)
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) $(filter %.cpp,$^) -o $@ $(LIBHALIDE_LDFLAGS)

.PRECIOUS: $(BIN)/%/matmul.a
$(BIN)/%/matmul.a: $(GENERATOR_BIN)/matmul.generator
	@mkdir -p $(@D)
	$^ -g matmul -e $(GENERATOR_OUTPUTS),conceptual_stmt_html -o $(@D) -f matmul target=$(HL_TARGET) gpu_schedule=$* M=1024 N=1024 K=1024

$(BIN)/%/matmul: process.cpp $(BIN)/%/matmul.a 
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) -I$(BIN)/$* -Wall $^ -o $@ $(LDFLAGS) $(IMAGE_IO_FLAGS) $(CUDA_LDFLAGS) $(OPENCL_LDFLAGS) -DBENCHMARK_HEADER="\"$(BIN)/$*/matmul.h\"" -DBENCHMARK_NAME="\"matmul\"" -DKERNEL_SIZE=$(KERNEL_SIZE) -DMATMUL_N=1024 -DMATMUL_M=1024 -DMATMUL_K=1024 -DSCHEDULE="\"$*\"" -DVERIFY_OUTPUT=true -DRUN_matmul
