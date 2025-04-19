# Compiler and flags
NVCC = nvcc
CFLAGS = -O3 -I../include
LDFLAGS = 

# Source files
FEATURE_EXTRACTION_SRC = feature_extraction.cu
STB_IMAGE_SRC = stb_image_impl.cu
TRAINING_SRC = training.cu
PREDICT_SRC = predict.cu

# Object files
FEATURE_EXTRACTION_OBJ = feature_extraction.o
STB_IMAGE_OBJ = stb_image_impl.o
TRAINING_OBJ = training.o
PREDICT_OBJ = predict.o

# Targets
TRAINING_TARGET = training
PREDICT_TARGET = predict

.PHONY: all clean

all: $(TRAINING_TARGET) $(PREDICT_TARGET)

# Compile feature extraction separately
$(FEATURE_EXTRACTION_OBJ): $(FEATURE_EXTRACTION_SRC)
	$(NVCC) $(CFLAGS) -c $< -o $@ -Xcompiler -fPIC

# Compile STB Image implementation separately
$(STB_IMAGE_OBJ): $(STB_IMAGE_SRC)
	$(NVCC) $(CFLAGS) -c $< -o $@ -Xcompiler -fPIC

# Link training with feature extraction and STB image
$(TRAINING_TARGET): $(TRAINING_OBJ) $(FEATURE_EXTRACTION_OBJ) $(STB_IMAGE_OBJ)
	$(NVCC) $(CFLAGS) $^ -o $@ $(LDFLAGS)

# Link predict with feature extraction and STB image
$(PREDICT_TARGET): $(PREDICT_OBJ) $(FEATURE_EXTRACTION_OBJ) $(STB_IMAGE_OBJ)
	$(NVCC) $(CFLAGS) $^ -o $@ $(LDFLAGS)

%.o: %.cu
	$(NVCC) $(CFLAGS) -c $< -o $@

clean:
	rm -f $(TRAINING_TARGET) $(PREDICT_TARGET) *.o
