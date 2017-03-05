#!/bin/sh
#
# ./build
# ./build clean
#

eb cxx="clang++" \
   flags="-std=c++11 -Wall -O3 -I /usr/local/include" \
   ldflags="-L /usr/local/lib -openmp -larmadillo" \
   workdir=work out=mnist $1
