CC = gcc
CFLAGS = -O2 -lm

all: classifier

classifier: classifier.c stb_image.h
	$(CC) $(CFLAGS) -o classifier classifier.c

clean:
	rm -f classifier