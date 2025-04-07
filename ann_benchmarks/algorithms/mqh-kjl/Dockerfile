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
RUN mkdir mqh-kjl && \
    mkdir -p mqh-kjl/external && \
    cd mqh-kjl/external && \
    git clone https://github.com/pybind/pybind11.git && \
    rm -rf pybind11/.git

# Copy the setup files from the algorithm directory
COPY ann_benchmarks/algorithms/mqh-kjl/binding/setup_mqh.py \
     ann_benchmarks/algorithms/mqh-kjl/binding/python_wrapper_mqh.cpp \
     ann_benchmarks/algorithms/mqh-kjl/binding/visited_list_pool.h \
     ann_benchmarks/algorithms/mqh-kjl/binding/mqh.h \
     /home/app/mqh-kjl/

# Build the module in /home/app
WORKDIR /home/app/mqh-kjl
RUN python3 setup_mqh.py build_ext --inplace
RUN cp pymqhkjl*.so /home/app/

# Test import from /home/app
WORKDIR /home/app
RUN python3 -c 'import pymqhkjl'

WORKDIR /home/app