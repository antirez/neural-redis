# tinydnnc

Minimal C bindings for [tiny-dnn](https://github.com/tiny-dnn/tiny-dnn).

**WARNING**: IN FLUX.

## Build

To build the library:
```
sh get_deps.sh
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE:STRING=Release ../ && make
```

To build the example (mnist):
```
cd example
make
```

## License

BSD license https://opensource.org/licenses/BSD-3-Clause

Copyright 2016, Luca Antiga, Orobix Srl (www.orobix.com).

