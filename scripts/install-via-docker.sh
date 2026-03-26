#!/bin/sh -l
set -e

# Usage: install-via-docker.sh <install-path> [sequre-src-path]
# Example: ./install-via-docker.sh $HOME/.sequre
# Example: ./install-via-docker.sh $HOME/.sequre /path/to/sequre
#
# Builds sequre inside Docker (manylinux2014) for ABI compatibility
# and extracts the result to the specified installation path.
#
# Options:
#   --rebuild    Force rebuild of Docker image (re-downloads Codon/LLVM/seq)

REBUILD_IMAGE=0
if [ "$1" = "--rebuild" ]; then
  REBUILD_IMAGE=1
  shift
fi

if [ -z "$1" ]; then
  echo "Usage: $0 [--rebuild] <install-path> [sequre-src-path]"
  echo "  install-path: Directory where Codon+Sequre will be installed"
  echo "  sequre-src-path: Path to sequre source (default: script's parent directory)"
  echo ""
  echo "Options:"
  echo "  --rebuild    Force rebuild of Docker image"
  exit 1
fi

INSTALL_PATH=$1

# Default to parent of scripts directory (i.e., the sequre repo root)
if [ -z "$2" ]; then
  SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
  SEQURE_SRC=$(dirname "$SCRIPT_DIR")
else
  SEQURE_SRC=$(cd "$2" && pwd)
fi

# Verify sequre source exists
if [ ! -f "$SEQURE_SRC/CMakeLists.txt" ]; then
  echo "Error: $SEQURE_SRC does not appear to be a valid sequre source directory"
  exit 1
fi

# Detect architecture
ARCH=$(uname -m)
case "$ARCH" in
  x86_64)  DOCKER_IMAGE="sequre-builder-x86_64" ;;
  aarch64) DOCKER_IMAGE="sequre-builder-aarch64" ;;
  *)
    echo "Error: Unsupported architecture: $ARCH"
    exit 1
    ;;
esac

# Build Docker image if it doesn't exist or --rebuild requested (caches Codon/LLVM/seq)
if [ "$REBUILD_IMAGE" = "1" ] || ! docker image inspect "$DOCKER_IMAGE" >/dev/null 2>&1; then
  echo "Building Docker image $DOCKER_IMAGE (this caches Codon/LLVM/seq for future builds)..."
  docker build \
    -t "$DOCKER_IMAGE" \
    -f "$SEQURE_SRC/docker/local-build/Dockerfile.$ARCH" \
    "$SEQURE_SRC/docker/local-build"
fi

echo "Building sequre using Docker ($DOCKER_IMAGE)..."
echo "  Source: $SEQURE_SRC"
echo "  Install to: $INSTALL_PATH"

# Create temp directory for output
BUILD_OUTPUT=$(mktemp -d)
trap "rm -rf $BUILD_OUTPUT" EXIT

# Run Docker build
# Copy source to writable location, build sequre, output tarball to /output
docker run --rm \
  -v "$SEQURE_SRC:/src:ro" \
  -v "$BUILD_OUTPUT:/output" \
  --entrypoint /bin/sh \
  "$DOCKER_IMAGE" \
  -c "
    cp -r /src /build/sequre && \
    rm -rf /build/sequre/build && \
    /build-sequre.sh /build/sequre
  "

# Find the generated tarball
TARBALL=$(ls "$BUILD_OUTPUT"/sequre-*.tar.gz 2>/dev/null | head -1)
if [ -z "$TARBALL" ]; then
  echo "Error: Build failed - no tarball generated"
  exit 1
fi

echo "Build complete. Extracting to $INSTALL_PATH..."

# Create install directory and extract
mkdir -p "$INSTALL_PATH"
tar xzvf "$TARBALL" -C "$INSTALL_PATH"

echo ""
echo "Done! Sequre installed to $INSTALL_PATH"
echo "Run with: $INSTALL_PATH/bin/sequre <file.codon>"
