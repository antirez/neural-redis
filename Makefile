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

all: neuralredis.so

.c.xo:
	$(CC) -I. $(CFLAGS) $(SHOBJ_CFLAGS) -fPIC -c $< -o $@

tinydnnc/build/tinydnnc.dylib:
	(cd tinydnnc; mkdir -p build; cd build; cmake -DCMAKE_BUILD_TYPE:STRING=Release ../ && make)

neuralredis.xo: redismodule.h tinydnnc/build/tinydnnc.dylib
neuralredis.so: neuralredis.xo
	$(LD) -o $@ $< $(SHOBJ_LDFLAGS) $(LIBS) -lc -ltinydnnc -L ./tinydnnc/build/

clean:
	rm -rf *.xo *.so
	(cd tinydnnc/build && make clean)
