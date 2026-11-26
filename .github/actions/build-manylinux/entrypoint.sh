#!/bin/sh -l
# Builds the Sequre release archive sequre-<os>-<arch>.tar.gz for the machine it runs on: inside the manylinux
# images on Linux (Dockerfile.x86_64, Dockerfile.aarch64), natively on macOS. Used by .github/workflows/ci.yml.
#
#   entrypoint.sh <sequre source directory>
#
# Steps, in this order (SEQURE_BUILD_STEPS selects some of them):
#   toolchain  download Codon and LLVM-codon (pinned versions below) into the prefixes
#   plugin     build the Sequre plugin with cmake and install it into <prefix>/lib/codon/plugins/sequre, with the
#              Codon version it is built for in codon-version (install.sh installs that Codon with the archive)
#   launcher   build the sequre launcher and, on x86-64 Linux, the alignment shim
#   gmp        bundle GMP built for the target: the repository's copy (external/GMP/lib: x86-64 Linux, arm64 macOS)
#              or, on other Linux architectures, one built from source (scripts/build_gmp.sh)
#   pack       write sequre-<os>-<arch>.tar.gz into the source directory
#   check      check that every binary in the archive is built for the target (scripts/check_machine_types.sh)
# Environment (defaults in brackets):
#   SEQURE_BUILD_PREFIX  the prefix the archive is assembled in [$HOME/.sequre]
#   SEQURE_LLVM_PREFIX   where LLVM-codon is unpacked [/opt]
#   SEQURE_BUILD_STEPS   [toolchain plugin launcher gmp pack check]
#   CC, CXX              compilers [macOS: the system clang; Linux: LLVM-codon's clang if unpacked, else cc, c++]
set -eu

SRC=$(cd "${1:?usage: entrypoint.sh <sequre source directory>}" && pwd)
PREFIX=${SEQURE_BUILD_PREFIX:-$HOME/.sequre}
OPT=${SEQURE_LLVM_PREFIX:-/opt}
STEPS=${SEQURE_BUILD_STEPS:-toolchain plugin launcher gmp pack check}
OS=$(uname -s | awk '{print tolower($0)}')
ARCH=$(uname -m)
PLATFORM=$OS-$ARCH
PLUGIN=$PREFIX/lib/codon/plugins/sequre
ARCHIVE=$SRC/sequre-$PLATFORM.tar.gz

CODON_VERSION=v0.20.8
LLVM_VERSION=20.1.7

has_step() {
  case " $STEPS " in
    *" $1 "*) return 0 ;;
  esac
  return 1
}

# download <url> <file>: curl -f, so that an HTTP error (a missing asset) stops the build here with the URL, instead
# of a tar error on the error page
download() {
  echo "Downloading $1"
  if ! curl -fL --retry 3 --retry-delay 2 -sS -o "$2" "$1"; then
    echo "error: could not download $1" >&2
    exit 1
  fi
}

TMP=$(mktemp -d "${TMPDIR:-/tmp}/sequre-build.XXXXXX")
trap 'rm -rf "$TMP"' EXIT

if has_step toolchain; then
  mkdir -p "$PREFIX/lib/codon/plugins" "$OPT"
  download "https://github.com/exaloop/codon/releases/download/$CODON_VERSION/codon-$PLATFORM.tar.gz" "$TMP/codon.tar.gz"
  download "https://github.com/exaloop/llvm-project/releases/download/codon-$LLVM_VERSION/llvm-codon-$LLVM_VERSION-$PLATFORM.tar.bz2" \
    "$TMP/llvm.tar.bz2"
  tar -xzf "$TMP/codon.tar.gz" -C "$PREFIX" --strip-components=1
  rm -f "$TMP/codon.tar.gz"
  # The LLVM tarball may be rooted at opt/ or at llvm-codon/; either way it lands at $OPT/llvm-codon
  LLVM_TOP=$(tar -tjf "$TMP/llvm.tar.bz2" | head -1 | cut -d/ -f1)
  if [ "$LLVM_TOP" = "opt" ]; then
    tar -xjf "$TMP/llvm.tar.bz2" -C "$OPT" --strip-components=1
  else
    tar -xjf "$TMP/llvm.tar.bz2" -C "$OPT"
  fi
  rm -f "$TMP/llvm.tar.bz2"
fi

# Pick the C/C++ compiler:
#  - macOS: Apple's system clang. It knows the SDK sysroot for both the plugin build and the standalone launcher.
#    (The LLVM-codon darwin tarball now ships a clang too, but it has no default sysroot, so system headers like
#    glob.h are not found when building the launcher directly.)
#  - Linux: the LLVM-codon clang, to match the bundled LLVM toolchain.
case "$OS" in
  darwin)
    CC=${CC:-$(command -v clang)}
    CXX=${CXX:-$(command -v clang++)}
    ;;
  *)
    if [ -f "$OPT/llvm-codon/bin/clang" ]; then
      CC=${CC:-$OPT/llvm-codon/bin/clang}
      CXX=${CXX:-$OPT/llvm-codon/bin/clang++}
    else
      CC=${CC:-cc}
      CXX=${CXX:-c++}
    fi
    ;;
esac

# On manylinux_2_28 the default toolchain is a gcc-toolset under /opt/rh, which the LLVM-codon clang does not
# auto-detect, so it cannot find libstdc++ (or the C++20 headers) when linking the plugin. Point clang at it via a
# config file, matching how Codon builds its own plugins (see exaloop/codon
# .github/build-linux/Dockerfile.linux-x86_64). No-op off manylinux: the glob is empty (e.g. on macOS), so no config
# file is written.
if [ -f "$OPT/llvm-codon/bin/clang" ]; then
  GCC_INSTALL_DIR=$(ls -d /opt/rh/gcc-toolset-*/root/usr/lib/gcc/*/* 2> /dev/null | sort -V | tail -1)
  if [ -n "$GCC_INSTALL_DIR" ]; then
    echo "--gcc-install-dir=$GCC_INSTALL_DIR" > "$OPT/llvm-codon/bin/clang.cfg"
    echo "--gcc-install-dir=$GCC_INSTALL_DIR" > "$OPT/llvm-codon/bin/clang++.cfg"
  fi
fi

if has_step plugin; then
  cmake -S "$SRC" -B "$SRC/build" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCODON_PATH="$PREFIX" \
    -DLLVM_DIR="$OPT/llvm-codon/lib/cmake/llvm" \
    -DCMAKE_C_COMPILER="$CC" \
    -DCMAKE_CXX_COMPILER="$CXX"
  cmake --build "$SRC/build" --target sequre
  cmake --install "$SRC/build" --prefix "$PLUGIN"
  echo "$CODON_VERSION" > "$PLUGIN/codon-version"
fi

# Sequre's numpy is a fork (its ndarray is defined differently and is incompatible with Codon's own; for now — the
# plan is to eventually migrate Sequre to use Codon's numpy directly). It is shipped inside the plugin (via the
# `install(DIRECTORY stdlib ...)` rule) and imported exclusively through the `sequre.stdlib.numpy` namespace, so it
# never collides with Codon's native `numpy`, which is left intact.

if has_step launcher; then
  mkdir -p "$PREFIX/bin" "$PREFIX/lib"
  "$CC" -O2 -o "$PREFIX/bin/sequre" "$SRC/sequre_launcher.c"
  # The x86-64 heap-vector over-align shim: the launcher preloads it for `sequre run` and links it into the binaries
  # of `sequre build`; install.sh patches Codon's Ptr separately (Codon's stdlib is not bundled here). ARM is
  # unaffected, so no shim is built or bundled there.
  if [ "$PLATFORM" = "linux-x86_64" ]; then
    "$CC" -shared -fPIC -O2 -o "$PREFIX/lib/sequre_align64.so" "$SRC/sequre_align64.c" -ldl
  fi
fi

if has_step gmp; then
  mkdir -p "$PLUGIN/lib"
  case "$PLATFORM" in
    darwin-arm64) cp "$SRC/external/GMP/lib/libgmp.dylib" "$PLUGIN/lib/libgmp.dylib" ;;
    linux-x86_64) cp "$SRC/external/GMP/lib/libgmp.so" "$PLUGIN/lib/libgmp.so" ;;
    linux-*) sh "$SRC/scripts/build_gmp.sh" "$PLUGIN/lib/libgmp.so" ;;
    *)
      echo "error: no GMP for $PLATFORM" >&2
      exit 1
      ;;
  esac
fi

if has_step pack; then
  # Only Sequre's own files (not the whole lib/codon/stdlib tree): install.sh unpacks Codon itself separately.
  # Sequre's numpy fork travels inside the plugin (lib/codon/plugins/sequre/stdlib/numpy), so Codon's native numpy/
  # is neither bundled nor replaced.
  case "$OS" in
    darwin) DLL=dylib ;;
    *) DLL=so ;;
  esac
  set -- bin/sequre lib/codon/plugins/sequre/plugin.toml lib/codon/plugins/sequre/build/libsequre.$DLL \
    lib/codon/plugins/sequre/lib/libgmp.$DLL lib/codon/plugins/sequre/stdlib/sequre/__init__.codon \
    lib/codon/plugins/sequre/codon-version
  if [ "$PLATFORM" = "linux-x86_64" ]; then
    set -- "$@" lib/sequre_align64.so
  fi
  for f in "$@"; do
    if [ ! -e "$PREFIX/$f" ]; then
      echo "error: $PREFIX/$f is missing: run the steps that make it first" >&2
      exit 1
    fi
  done
  set -- bin/sequre lib/codon/plugins/sequre
  if [ "$PLATFORM" = "linux-x86_64" ]; then
    set -- "$@" lib/sequre_align64.so
  fi
  tar -czf "$ARCHIVE" -C "$PREFIX" "$@"
  echo "Packed $ARCHIVE"
fi

if has_step check; then
  sh "$SRC/scripts/check_machine_types.sh" "$PLATFORM" "$ARCHIVE"
fi
echo "Done"
