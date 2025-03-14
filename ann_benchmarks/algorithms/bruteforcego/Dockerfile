FROM ann-benchmarks
# Add this near the top of your Dockerfile
ARG CACHEBUST=$(date +%s)

# Install Go 1.19 (which has the slices package)
RUN apt-get update && apt-get install -y golang-1.23
ENV PATH=/usr/lib/go-1.23/bin:$PATH

# Set up Go environment
ENV GOPATH=/go
ENV PATH=$GOPATH/bin:/usr/local/go/bin:$PATH
RUN mkdir -p $GOPATH/src

# Clone your algorithm repository
WORKDIR /app
RUN git clone https://github.com/incredible7/MQH_thesis.git
RUN cp -r /app/MQH_thesis $GOPATH/src/

# Fix module and imports
WORKDIR $GOPATH/src/MQH_thesis
RUN rm -f go.mod go.sum
RUN go mod init github.com/incredible7/MQH_thesis
# Tell Go to use the local directory for this module
RUN go mod edit -replace github.com/incredible7/MQH_thesis=./
# Fix import paths in all Go files
RUN find . -type f -name "*.go" -exec sed -i 's|MQH_THESIS/|github.com/incredible7/MQH_thesis/|g' {} \;

# Install dependencies
RUN go get -v ./...

# Copy your local binding directory into the container
COPY ann_benchmarks/algorithms/bruteforcego/binding /app/ann_benchmarks/algorithms/bruteforcego/binding

# Then proceed with the build
WORKDIR /app/ann_benchmarks/algorithms/bruteforcego/binding
RUN go build -buildmode=c-shared -o libgowrapper.so gowrapper.go || (echo "Build failed" && exit 1)

# Verify the library was created
RUN ls -la libgowrapper.so
