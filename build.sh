#!/bin/sh
#
# ./build
# ./build clean
#

sb cxx="clang++" \
   cxxflags="-std=c++17 -Wall -I /usr/local/include" \
   ldflags="-L /usr/local/lib -larmadillo -lopenblas" \
   workdir=work target=mnist $1
