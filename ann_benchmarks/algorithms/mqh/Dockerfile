FROM ann-benchmarks

# Install required packages
RUN apt-get update && apt-get install -y \
    build-essential \
    g++ \
    gcc \
    make \
    python3 \
    python3-pip \
    software-properties-common \
    wget \
    git \
    libtbb-dev \
    && rm -rf /var/lib/apt/lists/*

# Install g++-11 and set it as the default g++ compiler
RUN add-apt-repository ppa:ubuntu-toolchain-r/test \
    && apt-get update \
    && apt-get install -y g++-11 gcc-11 \
    && update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-11 100 \
    && update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-11 100 \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages
RUN pip3 install numpy scipy matplotlib pybind11

WORKDIR /home/app

# Now clone the BC-Tree repo for the 'balltree' algorithm import
# Clone BC-Tree repository
RUN git clone https://github.com/HuangQiang/BC-Tree.git && \
    rm -rf BC-Tree/.git && \
    mkdir -p BC-Tree/external && \
    cd BC-Tree/external && \
    git clone https://github.com/pybind/pybind11.git && \
    rm -rf pybind11/.git && \
    cd .. && \
    rm -f methods/ball_node.h methods/ball_tree.h

# Copy the setup files from the algorithm directory
COPY ann_benchmarks/algorithms/mqh/binding/setup_bt.py \
     ann_benchmarks/algorithms/mqh/binding/python_wrapper_bt.cpp \
     ann_benchmarks/algorithms/mqh/binding/CMakeLists.txt \
     ann_benchmarks/algorithms/mqh/binding/ball_node.h \
     ann_benchmarks/algorithms/mqh/binding/ball_tree.h \
     /home/app/BC-Tree/

# Move the ball_*.h files to the correct location
RUN cd /home/app/BC-Tree && \
    mv ball_node.h ball_tree.h methods/

# Build the module in /home/app
WORKDIR /home/app/BC-Tree
RUN python3 setup_bt.py build_ext --inplace
RUN cp b_tree*.so /home/app/

# Test import from /home/app
WORKDIR /home/app
RUN python3 -c 'import b_tree'

WORKDIR /home/app

# Now clone the P2HNNS for the 'mh' algorithm import
# Clone P2HNNS repository
RUN git clone https://github.com/HuangQiang/P2HNNS.git && \
    rm -rf P2HNNS/.git && \
    mkdir -p P2HNNS/external && \
    cd P2HNNS/external && \
    git clone https://github.com/pybind/pybind11.git && \
    rm -rf pybind11/.git && \
    cd .. && \
    rm -f methods/baseline.h methods/mh.h methods/bh.h methods/kl_bucketing.h

# Copy the setup files from the algorithm directory
COPY ann_benchmarks/algorithms/mqh/binding/setup_mh.py \
     ann_benchmarks/algorithms/mqh/binding/python_wrapper_mh.cpp \
     ann_benchmarks/algorithms/mqh/binding/CMakeLists.txt \
     ann_benchmarks/algorithms/mqh/binding/baseline.h \
     ann_benchmarks/algorithms/mqh/binding/mh.h \
     ann_benchmarks/algorithms/mqh/binding/bh.h \
     ann_benchmarks/algorithms/mqh/binding/kl_bucketing.h \
     /home/app/P2HNNS/

# Build the module in /home/app
WORKDIR /home/app/P2HNNS
RUN python3 setup_mh.py build_ext --inplace
RUN cp mh*.so /home/app/

# Test import from /home/app
WORKDIR /home/app
RUN python3 -c 'import mh'

RUN mkdir mqh && \
    mkdir -p mqh/external && \
    cd mqh/external && \
    git clone https://github.com/pybind/pybind11.git && \
    rm -rf pybind11/.git

# Copy the setup files from the algorithm directory
COPY ann_benchmarks/algorithms/mqh/binding/setup_mqh.py \
     ann_benchmarks/algorithms/mqh/binding/python_wrapper_mqh.cpp \
     ann_benchmarks/algorithms/mqh/binding/visited_list_pool.h \
     ann_benchmarks/algorithms/mqh/binding/mqh.h \
     /home/app/mqh/

# Build the module in /home/app
WORKDIR /home/app/mqh
RUN python3 setup_mqh.py build_ext --inplace
RUN cp pymqh*.so /home/app/

# Test import from /home/app
WORKDIR /home/app
RUN python3 -c 'import pymqh'

WORKDIR /home/app