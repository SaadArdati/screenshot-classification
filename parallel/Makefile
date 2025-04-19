NVCC = nvcc
NVCC_FLAGS = -O3 -I../include

# Source files
TRAINING_MAIN = training.cu
PREDICT_MAIN = predict.cu

# Output executables
TRAINING_EXEC = training
PREDICT_EXEC = predict

all: $(TRAINING_EXEC) $(PREDICT_EXEC)

$(TRAINING_EXEC): $(TRAINING_MAIN)
	$(NVCC) $(NVCC_FLAGS) -o $@ $^

$(PREDICT_EXEC): $(PREDICT_MAIN)
	$(NVCC) $(NVCC_FLAGS) -o $@ $^

clean:
	rm -f $(TRAINING_EXEC) $(PREDICT_EXEC) 