all: nn-test-1 nn-test-2 nn-benchmark

nn-test-1: nn-test-1.c ../nn.c ../nn.h
	$(CC) nn-test-1.c ../nn.c -Wall -W -O2 -o nn-test-1

nn-test-2: nn-test-2.c ../nn.c ../nn.h
	$(CC) nn-test-2.c ../nn.c -Wall -W -O2 -o nn-test-2

nn-benchmark: nn-benchmark.c ../nn.c ../nn.h
	$(CC) nn-benchmark.c ../nn.c -Wall -W -O2 -o nn-benchmark
