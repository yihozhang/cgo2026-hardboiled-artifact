HL_TARGET ?= host-cuda

include ../support/Makefile.inc

KERNEL_SIZE ?= 16

all: $(BIN)/cuda_only/conv1d $(BIN)/tensorcore/conv1d

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
