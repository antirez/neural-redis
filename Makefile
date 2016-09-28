# find the OS
uname_S := $(shell sh -c 'uname -s 2>/dev/null || echo not')

# Compile flags for linux / osx
ifeq ($(uname_S),Linux)
	SHOBJ_CFLAGS ?=  -fno-common -g -ggdb -std=c99
	SHOBJ_LDFLAGS ?= -shared
else
	SHOBJ_CFLAGS ?= -dynamic -fno-common -g -ggdb -std=c99
	SHOBJ_LDFLAGS ?= -bundle -undefined dynamic_lookup
endif

.SUFFIXES: .c .so .xo .o

all: neuralredis.so testmodule.so

.c.xo:
	$(CC) -I. $(CFLAGS) $(SHOBJ_CFLAGS) -fPIC -c $< -o $@

neuralredis.xo: redismodule.h

neuralredis.so: neuralredis.xo nn.xo
	$(LD) -o $@ $< nn.xo $(SHOBJ_LDFLAGS) $(LIBS) -lc

clean:
	rm -rf *.xo *.so
