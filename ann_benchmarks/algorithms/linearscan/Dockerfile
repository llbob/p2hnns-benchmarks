FROM ann-benchmarks

# Install dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    g++ \
    gcc \
    make \
    cmake \
    libboost-all-dev \
    software-properties-common \
    && rm -rf /var/lib/apt/lists/*

# Install g++-11 and set it as the default g++ compiler (instead of g++-8)
RUN add-apt-repository ppa:ubuntu-toolchain-r/test \
    && apt-get update \
    && apt-get install -y g++-11 gcc-11 \
    && update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-11 100 \
    && update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-11 100 \
    && ln -s /usr/bin/g++ /usr/bin/g++-8 \
    && rm -rf /var/lib/apt/lists/*

# Clone the P2H repository
WORKDIR /home/app
RUN git clone https://github.com/HuangQiang/P2HNNS.git p2h-ann

# Build the P2H code
WORKDIR /home/app/p2h-ann/methods
RUN make clean && make -j

# Copy our module - specify the correct path
COPY ann_benchmarks/algorithms/linearscan/module.py /home/app/

# Set environment variables properly without warning
ENV LD_LIBRARY_PATH=/home/app/p2h-ann/methods