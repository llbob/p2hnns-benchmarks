FROM ann-benchmarks

# Install required packages for C++ compilation with OpenMP support
RUN apt-get update && apt-get install -y \
    build-essential \
    g++ \
    libomp-dev \
    make \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /home/app

# Copy the algorithm code
COPY . /home/app/

# Build the Ball-Tree implementation
RUN cd external_repo/methods && \
    g++ -fopenmp -O3 -o p2h_search main.cc util.cc pri_queue.cc qalsh.cc rqalsh.cc

# Create a wrapper script to run the algorithm on CIFAR
RUN echo '#!/bin/bash\n\
cd /home/app/external_repo/methods\n\
./p2h_search -alg 2 -n 50000 -qn 100 -d 513 -leaf 1000 -cf config -dt float32 \
-dn Cifar -ds /home/app/data/bin/cifar/Cifar.ds -qs /home/app/data/bin/cifar/Cifar.q \
-ts /home/app/data/bin/cifar/Cifar.gt -op /home/app/results/\n\
' > /home/app/run_balltree.sh && chmod +x /home/app/run_balltree.sh

# Override the entrypoint to use our wrapper script
ENTRYPOINT ["/home/app/run_balltree.sh"]