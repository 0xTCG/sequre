FROM hsmile/eva:latest
LABEL Name=seal

# Copy files
COPY applications/others/seal/ SEAL/native/examples/
WORKDIR /SEAL/native/examples/

RUN \
    cmake -S . -B build && \
    cmake --build build

# Run benchmarks
CMD ["./build/bin/sealexamples"]
