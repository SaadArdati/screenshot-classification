NVCC = nvcc
NVCC_FLAGS = -O3 -I../include

# Source files
COMMON_SOURCES = feature_extraction.cu knn.cu
TRAINING_MAIN = training.cu
PREDICT_MAIN = predict.cu

# Output executables
TRAINING_EXEC = training
PREDICT_EXEC = predict

all: $(TRAINING_EXEC) $(PREDICT_EXEC)

$(TRAINING_EXEC): $(TRAINING_MAIN) $(COMMON_SOURCES)
	$(NVCC) $(NVCC_FLAGS) -o $@ $^

$(PREDICT_EXEC): $(PREDICT_MAIN) $(COMMON_SOURCES)
	$(NVCC) $(NVCC_FLAGS) -o $@ $^

clean:
	rm -f $(TRAINING_EXEC) $(PREDICT_EXEC) 