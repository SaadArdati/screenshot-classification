CC = gcc
CFLAGS = -O3 -Wall -I../include
LIBS = -lm

all: train_seq predict_seq

train_seq: training.c
	$(CC) $(CFLAGS) -o train_seq training.c $(LIBS)

predict_seq: predict.c
	$(CC) $(CFLAGS) -o predict_seq predict.c $(LIBS)

clean:
	rm -f train_seq predict_seq *.o

.PHONY: all clean