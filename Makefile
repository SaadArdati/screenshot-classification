CC = gcc
CFLAGS = -O2 -lm

all: training predict

training: training.c stb_image.h
	$(CC) $(CFLAGS) -o training training.c

predict: predict.c stb_image.h
	$(CC) $(CFLAGS) -o predict predict.c

clean:
	rm -f training predict