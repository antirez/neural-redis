# find the OS
uname_S := $(shell sh -c 'uname -s 2>/dev/null || echo not')

# Compile flags for linux / osx
ifeq ($(uname_S),Linux)
	SHOBJ_CFLAGS ?= -Wall -W -O3 -fno-common -g -ggdb -std=c99
	SHOBJ_LDFLAGS ?= -shared
else
	SHOBJ_CFLAGS ?= -Wall -W -O3 -dynamic -fno-common -g -ggdb -std=c99
	SHOBJ_LDFLAGS ?= -bundle -undefined dynamic_lookup
endif

.SUFFIXES: .c .so .xo .o

all:
	@echo ""
	@echo "Make neon     -- Faster if you have a modern ARM CPU."
	@echo "Make sse     -- Faster if you have a modern CPU."
	@echo "Make avx     -- Even faster if you have a modern CPU."
	@echo "Make generic -- Works everywhere."
	@echo ""
	@echo "The avx code uses AVX2, it requires Haswell (Q2 2013) or better."
	@echo ""

generic: neuralredis.so
neon:
	make neuralredis.so CFLAGS=-DUSE_NEON

sse:
	make neuralredis.so CFLAGS=-DUSE_SSE SSE="-msse3"

avx:
	make neuralredis.so CFLAGS=-DUSE_AVX AVX="-mavx2 -mfma"

.c.xo:
	$(CC) -I. $(CFLAGS) $(SHOBJ_CFLAGS) $(AVX) $(SSE) -fPIC -c $< -o $@

nn.c: nn.h

neuralredis.xo: redismodule.h

neuralredis.so: neuralredis.xo nn.xo
	$(LD) -o $@ $< nn.xo $(SHOBJ_LDFLAGS) $(LIBS) -lc

clean:
	rm -rf *.xo *.so
