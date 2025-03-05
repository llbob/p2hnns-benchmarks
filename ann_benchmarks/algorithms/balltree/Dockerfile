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

# Find OpenMP headers in the container
RUN find / -name omp.h 2>/dev/null > omp_locations.txt || echo "omp.h not found"
RUN if [ -s omp_locations.txt ]; then \
      INCLUDE_PATH=$(head -n 1 omp_locations.txt | xargs dirname); \
      echo "Found omp.h at ${INCLUDE_PATH}"; \
      cd external_repo/methods && \
      g++ -fopenmp -I${INCLUDE_PATH} -O3 -o p2h_search main.cc util.cc pri_queue.cc qalsh.cc rqalsh.cc; \
    else \
      echo "Trying default paths"; \
      cd external_repo/methods && \
      g++ -fopenmp -O3 -o p2h_search main.cc util.cc pri_queue.cc qalsh.cc rqalsh.cc; \
    fi

# Use the standard entrypoint from the base image
ENTRYPOINT ["python3", "-u", "run_algorithm.py"]