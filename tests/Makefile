all: nn-test-1 nn-test-2 nn-benchmark

nn-test-1: nn-test-1.c ../nn.c ../nn.h
	$(CC) nn-test-1.c ../nn.c -Wall -W -O2 -o nn-test-1

nn-test-2: nn-test-2.c ../nn.c ../nn.h
	$(CC) nn-test-2.c ../nn.c -Wall -W -O2 -o nn-test-2

nn-benchmark: nn-benchmark.c ../nn.c ../nn.h
	$(CC) -DUSE_SSE nn-benchmark.c ../nn.c -Wall -W -O3 -o nn-benchmark

clean:
	rm -f nn-test-1 nn-test-2 nn-benchmark
