FROM golang:1.21.6-bullseye

# Install dependencies
RUN \
    apt-get update && \
    apt-get -y install python3-pip && \
    pip3 install toml-cli

COPY applications/others/lattigo_king /lattigo_king
COPY deps/sfgwas-private /sfgwas-private
COPY data/king/king_sample.txt /data/king_sample.txt

# Run benchmarks
WORKDIR /lattigo_king
CMD ["bash", "batch_run_basic.sh"]
