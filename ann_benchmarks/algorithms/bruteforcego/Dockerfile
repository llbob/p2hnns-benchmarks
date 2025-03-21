FROM ann-benchmarks

# Install required packages
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    wget \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Go for ARM64 architecture
RUN wget https://golang.org/dl/go1.22.0.linux-arm64.tar.gz && \
    tar -C /usr/local -xzf go1.22.0.linux-arm64.tar.gz && \
    rm go1.22.0.linux-arm64.tar.gz

# Add Go to PATH
ENV PATH=$PATH:/usr/local/go/bin
ENV GOPATH=/go
ENV PATH=$PATH:$GOPATH/bin

WORKDIR /home/app

# Clone the Go repository that contains algorithms, types, and utils
RUN git clone https://github.com/incredible7/MQH_thesis.git /go/src/MQH_thesis

# Create the binding directory structure
RUN mkdir -p /home/app/ann_benchmarks/algorithms/bruteforcego/binding

# Copy the gowrapper file from the local directory to the binding directory
COPY ann_benchmarks/algorithms/bruteforcego/binding/gowrapper.go /home/app/ann_benchmarks/algorithms/bruteforcego/binding/

# Copy the Python binding file to the module location
COPY ann_benchmarks/algorithms/bruteforcego/module.py /home/app/ann_benchmarks/algorithms/bruteforcego/

COPY ann_benchmarks/algorithms/base/module.py /home/app/ann_benchmarks/algorithms/base/

# Set up Go module and build
WORKDIR /home/app/ann_benchmarks/algorithms/bruteforcego/binding
RUN echo 'module bruteforcego \ngo 1.22 \nreplace MQH_thesis => /go/src/MQH_thesis' > go.mod

# Show architecture information
RUN uname -a && gcc --version

# Build for ARM architecture
RUN go mod tidy && \
    CGO_ENABLED=1 GOARCH=arm64 go build -buildmode=c-shared -o libgowrapper.so gowrapper.go

# Return to app directory
WORKDIR /home/app

# Test the import - using the correct module and class
RUN python3 -c 'import sys; sys.path.insert(0, "/home/app"); from ann_benchmarks.algorithms.bruteforcego.module import BruteForceGo; print("Import successful")'