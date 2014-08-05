#!/bin/sh
#
# ./build
# ./build clean
#

eb cxx="clang++" \
   flag="-std=c++11 -Wall -O3 -I /usr/local/include" \
   ldflag="-L /usr/local/lib -openmp -L /usr/local/lib/gcc47/ -lquadmath -larmadillo" \
   workdir=work out=mnist $1
