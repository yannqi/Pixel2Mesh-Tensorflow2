#!/bin/bash
nvcc=/usr/local/cuda-11.0/bin/nvcc
cudalib=/usr/local/cuda-11.0/lib64
TF_CFLAGS=( $(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))') )
TF_LFLAGS=( $(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))') )

#因为Makefile文件的include问题，我将其放到了sh文件内执行。效果不变
'/usr/local/cuda-11.0/bin/nvcc' -D_GLIBCXX_USE_CXX11_ABI=0 -std=c++11 -c -o tf_nndistance_g.cu.o tf_nndistance_g.cu ${TF_CFLAGS[@]} ${TF_LFLAGS[@]} -DGOOGLE_CUDA=1 -x cu -Xcompiler -fPIC -O2
g++ -std=c++11 tf_nndistance.cpp tf_nndistance_g.cu.o -o tf_nndistance_so.so -shared -fPIC ${TF_CFLAGS[@]} -lcudart ${TF_LFLAGS[@]} -L '/usr/local/cuda-11.0/lib64' -O2 -D_GLIBCXX_USE_CXX11_ABI=0
