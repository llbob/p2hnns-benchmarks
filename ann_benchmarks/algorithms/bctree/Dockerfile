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

# Install g++-11 and set it as the default g++ compiler (instead of g++-8)
RUN add-apt-repository ppa:ubuntu-toolchain-r/test \
    && apt-get update \
    && apt-get install -y g++-11 gcc-11 \
    && update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-11 100 \
    && update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-11 100 \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages for visualization and pybind11
RUN pip3 install numpy scipy matplotlib pybind11

# Set the working directory
WORKDIR /home/app

# Clone the repository and copy files
RUN git clone https://github.com/HuangQiang/BC-Tree.git && \
    rm -rf BC-Tree/.git && \
    mkdir -p BC-Tree/external && \
    cd BC-Tree/external && \
    git clone https://github.com/pybind/pybind11.git && \
    rm -rf pybind11/.git

# Copy the setup files from the algorithm directory
COPY ann_benchmarks/algorithms/bctree/setup.py \
     ann_benchmarks/algorithms/bctree/python_wrapper.cpp \
     ann_benchmarks/algorithms/bctree/CMakeLists.txt \
     /home/app/BC-Tree/

# Install the wrapper
WORKDIR /home/app/BC-Tree
RUN python3 setup.py build_ext --inplace