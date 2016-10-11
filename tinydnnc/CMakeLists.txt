
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories(deps/tiny-dnn)

add_library(tinydnnc SHARED tinydnnc.cc)

