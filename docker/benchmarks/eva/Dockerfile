FROM hsmile/eva:latest
LABEL Name=eva

# Copy files
COPY applications/others/eva/ EVA/examples/

# Run benchmarks
CMD ["sh", "EVA/examples/run.sh"]
