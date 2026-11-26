#!/bin/sh
# Builds GMP's shared library from its release source and copies it to the given path. The release build bundles the
# result where the repository has no copy for the target (Linux aarch64; external/GMP/lib holds the x86-64 Linux and
# the arm64 macOS libraries).
#
#   scripts/build_gmp.sh <output path, e.g. .../plugins/sequre/lib/libgmp.so>
#
# The source archive's SHA-256 is pinned. configure gets a generic CPU type for the machine's architecture, so the
# library runs on every CPU of it, not only on CPUs like the build machine's; `make check` runs GMP's test suite.
# Environment (optional):
#   GMP_SOURCE  a local copy of gmp-<version>.tar.xz to use instead of the download (checked against the same hash)
#   GMP_JOBS    parallel make jobs (default: the number of CPUs)
set -eu

GMP_VERSION=6.3.0
GMP_SHA256=a3c2b80201b89e68616f4ad30bc66aee4927c3ce50e33929ca819d5c43538898
GMP_URLS="https://ftp.gnu.org/gnu/gmp/gmp-$GMP_VERSION.tar.xz https://ftpmirror.gnu.org/gmp/gmp-$GMP_VERSION.tar.xz"

if [ $# -ne 1 ]; then
  echo "usage: $0 <output path of libgmp.so>" >&2
  exit 2
fi
case "$1" in
  /*) out=$1 ;;
  *) out=$PWD/$1 ;;
esac
jobs=${GMP_JOBS:-$(getconf _NPROCESSORS_ONLN 2> /dev/null || echo 2)}

case "$(uname -s)-$(uname -m)" in
  Linux-aarch64 | Linux-arm64) build=aarch64-unknown-linux-gnu ;;
  Linux-x86_64) build=x86_64-pc-linux-gnu ;;
  *)
    echo "error: build_gmp.sh builds for Linux on x86_64 or aarch64, not $(uname -s) $(uname -m)" >&2
    exit 1
    ;;
esac

missing=
tools="xz m4 make cc"
[ -n "${GMP_SOURCE:-}" ] || tools="curl $tools"
for tool in $tools; do
  command -v "$tool" > /dev/null 2>&1 || missing="$missing $tool"
done
if [ -n "$missing" ]; then
  echo "error: building GMP needs:$missing" >&2
  exit 1
fi

work=$(mktemp -d "${TMPDIR:-/tmp}/build-gmp.XXXXXX")
trap 'rm -rf "$work"' EXIT
tarball=$work/gmp-$GMP_VERSION.tar.xz

if [ -n "${GMP_SOURCE:-}" ]; then
  cp "$GMP_SOURCE" "$tarball"  # resolved before any cd
else
  for url in $GMP_URLS; do
    echo "Downloading GMP $GMP_VERSION: $url"
    if curl -fsSL --retry 3 --retry-delay 2 -o "$tarball" "$url"; then
      break
    fi
    rm -f "$tarball"
  done
  if [ ! -f "$tarball" ]; then
    echo "error: could not download GMP $GMP_VERSION from any of: $GMP_URLS" >&2
    exit 1
  fi
fi

if command -v sha256sum > /dev/null 2>&1; then
  sum=$(sha256sum "$tarball" | cut -d' ' -f1)
else
  sum=$(shasum -a 256 "$tarball" | cut -d' ' -f1)
fi
if [ "$sum" != "$GMP_SHA256" ]; then
  echo "error: gmp-$GMP_VERSION.tar.xz has SHA-256 $sum, expected $GMP_SHA256" >&2
  exit 1
fi

tar -xJf "$tarball" -C "$work"
cd "$work/gmp-$GMP_VERSION"
echo "Building GMP $GMP_VERSION for $build (generic CPU) with $jobs jobs ..."
if ! ./configure --enable-shared --disable-static --build="$build" --host="$build" > "$work/configure.log" 2>&1; then
  tail -40 "$work/configure.log" >&2
  exit 1
fi
make -j"$jobs" > "$work/make.log" 2>&1 || { tail -40 "$work/make.log" >&2; exit 1; }
make -j"$jobs" check > "$work/check.log" 2>&1 || { grep -E "FAIL|Error" "$work/check.log" | head -40 >&2; exit 1; }
echo "GMP $GMP_VERSION: $(grep -c '^PASS' "$work/check.log") tests passed"

mkdir -p "$(dirname "$out")"
cp -L .libs/libgmp.so "$out"
echo "Built $out"
