#!/bin/sh -l
set -e

# Usage: install-via-docker.sh <install-path> [sequre-src-path]
# Example: ./install-via-docker.sh $HOME/.sequre
# Example: ./install-via-docker.sh $HOME/.sequre /path/to/sequre
#
# Builds sequre inside Docker (manylinux_2_28) for ABI compatibility
# and extracts the result to the specified installation path. Linux only:
# the image builds Linux binaries (on macOS, use scripts/install.sh).
#
# Options:
#   --rebuild    Force rebuild of Docker image (re-downloads Codon/LLVM)

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

# The directories the install replaces lie under these; through a symbolic link, replacing them would change the
# link's target, outside the install
for d in lib lib/codon lib/codon/plugins include; do
  if [ -L "$INSTALL_PATH/$d" ]; then
    echo "Error: $INSTALL_PATH/$d is a symbolic link (to $(readlink "$INSTALL_PATH/$d")): the install replaces"
    echo "Codon's and Sequre's directories under it, which would change the link's target. Install into another"
    echo "directory, or replace the link with a copy of what it points to."
    exit 1
  fi
done

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

# The image builds Linux binaries: on macOS they would land in the prefix and not run there
if [ "$(uname -s)" != "Linux" ]; then
  echo "Error: install-via-docker.sh builds Linux binaries and installs them on Linux only."
  echo "On $(uname -s), install the release with scripts/install.sh, or build from source"
  echo "(docs/getting-started/build-from-source.md)."
  exit 1
fi

# Detect architecture
ARCH=$(uname -m)
case "$ARCH" in
  x86_64)        DOCKERFILE_ARCH="x86_64" ;;
  aarch64|arm64) DOCKERFILE_ARCH="aarch64" ;;
  *)
    echo "Error: Unsupported architecture: $ARCH"
    exit 1
    ;;
esac
# The image caches the Codon release its Dockerfile names and the build script it copies, so the name carries that
# release and a digest of the two files: an image built from others (before a Codon upgrade or a change to the build)
# is not reused
DOCKERFILE="$SEQURE_SRC/docker/local-build/Dockerfile.$DOCKERFILE_ARCH"
CODON_TAG=$(grep -o 'releases/download/v[0-9.]*' "$DOCKERFILE" | head -n 1 | sed 's|.*/||')
BUILD_DIGEST=$(cat "$DOCKERFILE" "$SEQURE_SRC/docker/local-build/build-sequre.sh" | sha256sum | cut -c1-12)
DOCKER_IMAGE="sequre-builder-$DOCKERFILE_ARCH${CODON_TAG:+-codon-${CODON_TAG#v}}-$BUILD_DIGEST"

# Build Docker image if it doesn't exist or --rebuild requested (caches Codon/LLVM)
if [ "$REBUILD_IMAGE" = "1" ] || ! docker image inspect "$DOCKER_IMAGE" >/dev/null 2>&1; then
  echo "Building Docker image $DOCKER_IMAGE (this caches Codon/LLVM for future builds)..."
  docker build \
    -t "$DOCKER_IMAGE" \
    -f "$SEQURE_SRC/docker/local-build/Dockerfile.$DOCKERFILE_ARCH" \
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

# Create the install directory and extract. As install.sh does, first replace what the tarball owns outright, so that
# an install over an earlier one keeps none of that one's files: Codon's stdlib and headers (the tarball carries no
# headers) and Sequre's plugin, which is moved to lib/codon/plugins/sequre.previous when it holds entries this build
# lacks (a user's files), but not for a lib/ of nothing but the shared libraries that earlier builds bundled there.
# The seq plugin that Sequre's archives for Codon 0.19 bundled goes too.
mkdir -p "$INSTALL_PATH"
tar -tzf "$TARBALL" > "$BUILD_OUTPUT/tarball.list"
in_tarball() {  # in_tarball <path>: the tarball has <path> or something under it
  awk -v p="$1" '{ sub(/^\.\//, ""); if ($0 == p || $0 == p "/" || index($0, p "/") == 1) { found = 1; exit } }
                 END { exit !found }' "$BUILD_OUTPUT/tarball.list"
}
only_shared_libraries() {  # only_shared_libraries <dir>: a directory of regular *.so, *.so.* or *.dylib files only
  [ -d "$1" ] && [ ! -L "$1" ] || return 1
  for f in "$1"/* "$1"/.[!.]* "$1"/..?*; do
    if [ -e "$f" ] || [ -L "$f" ]; then
      case "${f##*/}" in
        *.so|*.so.*|*.dylib) [ -f "$f" ] && [ ! -L "$f" ] || return 1 ;;
        *) return 1 ;;
      esac
    fi
  done
}
PLUGINS="$INSTALL_PATH/lib/codon/plugins"
rm -rf "$INSTALL_PATH/lib/codon/stdlib" "$INSTALL_PATH/include/codon"
if [ -d "$PLUGINS/seq" ] && [ -d "$PLUGINS/sequre" ] && [ ! -f "$PLUGINS/sequre/codon-version" ] &&
   ! in_tarball lib/codon/plugins/seq; then
  echo "Removing the seq plugin that the earlier Sequre release bundled (built for Codon 0.19)."
  rm -rf "$PLUGINS/seq"
fi
if [ -e "$PLUGINS/sequre" ] || [ -L "$PLUGINS/sequre" ]; then
  extra=""
  for entry in "$PLUGINS/sequre"/* "$PLUGINS/sequre"/.[!.]* "$PLUGINS/sequre"/..?*; do
    if [ -e "$entry" ] || [ -L "$entry" ]; then
      # lib: the shared libraries that earlier builds bundled beside the plugin, which this one does not: the build's
      # own, not the user's, as long as it holds nothing else
      if [ "${entry##*/}" = lib ] && ! in_tarball lib/codon/plugins/sequre/lib && only_shared_libraries "$entry"; then
        continue
      fi
      in_tarball "lib/codon/plugins/sequre/${entry##*/}" || extra="$extra ${entry##*/}"
    fi
  done
  if [ -n "$extra" ]; then
    rm -rf "$PLUGINS/sequre.previous"
    mv "$PLUGINS/sequre" "$PLUGINS/sequre.previous"
    echo "Moved the earlier Sequre plugin to $PLUGINS/sequre.previous: it holds entries this build lacks (${extra# })."
  else
    rm -rf "$PLUGINS/sequre"
  fi
fi
tar xzvf "$TARBALL" -C "$INSTALL_PATH"

# The x86-64 alignment workaround: the shim and the launcher are already in the tarball, so this only rebuilds the
# shim if its source is newer. No-op off x86_64.
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
if [ -f "$SCRIPT_DIR/apply_avx512_workaround.sh" ]; then
  SEQURE_WORKAROUND_SKIP_LAUNCHER=1 bash "$SCRIPT_DIR/apply_avx512_workaround.sh" "$INSTALL_PATH"
fi

echo ""
echo "Done! Sequre installed to $INSTALL_PATH"
echo "Run with: $INSTALL_PATH/bin/sequre <file.codon>"
