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

# Clone BC-Tree repository
RUN git clone https://github.com/HuangQiang/BC-Tree.git && \
    rm -rf BC-Tree/.git && \
    mkdir -p BC-Tree/external && \
    cd BC-Tree/external && \
    git clone https://github.com/pybind/pybind11.git && \
    rm -rf pybind11/.git

# Copy the setup files from the algorithm directory
COPY ann_benchmarks/algorithms/balltree/binding/setup.py \
     ann_benchmarks/algorithms/balltree/binding/python_wrapper_bt.cpp \
     ann_benchmarks/algorithms/balltree/binding/CMakeLists.txt \
     /home/app/BC-Tree/

# Build the module in /home/app
WORKDIR /home/app/BC-Tree
RUN python3 setup.py build_ext --inplace
RUN cp bc_tree*.so /home/app/

# Test import from /home/app
WORKDIR /home/app
RUN python3 -c 'import bc_tree'