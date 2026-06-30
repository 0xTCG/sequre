#!/usr/bin/env bash
set -e
set -o pipefail

SEQURE_INSTALL_DIR=~/.sequre
OS=$(uname -s | awk '{print tolower($0)}')
ARCH=$(uname -m)

if [ "$OS" = "linux" ]; then
  if [ "$ARCH" != "x86_64" ] && [ "$ARCH" != "aarch64" ]; then
    echo "error: Pre-built binaries for Linux only exist for x86_64 and aarch64." >&2
    exit 1
  fi
elif [ "$OS" = "darwin" ]; then
  if [ "$ARCH" != "arm64" ]; then
    echo "error: Pre-built binaries for macOS only exist for Apple Silicon (arm64)." >&2
    exit 1
  fi
else
  echo "error: Pre-built binaries only exist for Linux (x86_64, aarch64) and macOS (arm64)." >&2
  exit 1
fi

CODON_VERSION=v0.19.6
CODON_BUILD_ARCHIVE=codon-$OS-$ARCH.tar.gz
SEQURE_BUILD_ARCHIVE=sequre-$OS-$ARCH.tar.gz

echo "Installing Sequre to $SEQURE_INSTALL_DIR ..."

mkdir -p "$SEQURE_INSTALL_DIR"
cd "$SEQURE_INSTALL_DIR"

# 1. Install Codon runtime
echo "Downloading Codon $CODON_VERSION ..."
curl -L "https://github.com/exaloop/codon/releases/download/$CODON_VERSION/$CODON_BUILD_ARCHIVE" | tar zxvf - --strip-components=1

# Sequre's numpy is a fork (its ndarray is defined differently and is
# incompatible with Codon's own). It ships inside the plugin and is imported
# exclusively via the `sequre.stdlib.numpy` namespace, so it no longer
# collides with Codon's native `numpy` — Codon's own numpy/ is left intact.

# 2. Install Sequre (plugin + launcher) on top
echo "Downloading Sequre ..."
curl -L "https://github.com/0xTCG/sequre/releases/latest/download/$SEQURE_BUILD_ARCHIVE" | tar zxvf - --strip-components=0

# 3. Apply the AVX-512 heap-vector alignment workaround for Codon (no-op off
#    x86_64). A fresh Codon install ships an unpatched stdlib, so re-apply the
#    Ptr unaligned-load patch here (the shim and patched launcher come from the
#    Sequre tarball; Sequre's own stdlib, incl. prg.codon, ships in the plugin).
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
if [ -f "$SCRIPT_DIR/apply_avx512_workaround.sh" ]; then
  bash "$SCRIPT_DIR/apply_avx512_workaround.sh" "$SEQURE_INSTALL_DIR"
fi

EXPORT_COMMAND="export PATH=$SEQURE_INSTALL_DIR/bin:\$PATH"
echo ""
echo "PATH export command:"
echo "  $EXPORT_COMMAND"


PROFILES=()
for f in ~/.zshenv ~/.zshrc ~/.zprofile ~/.bash_profile ~/.bash_login ~/.bashrc ~/.profile; do
  if [ -e "$f" ]; then
    if ! grep -F -q "$EXPORT_COMMAND" "$f"; then
      PROFILES+=("$f")
    else
      echo "PATH already updated in $f; skipping."
    fi
  fi
done

if [ ${#PROFILES[@]} -eq 0 ]; then
  echo "No shell configuration files found to update PATH."
else
  echo "The following profile files will be updated:"
  for f in "${PROFILES[@]}"; do echo "  $f"; done
  read -p "Update PATH in the above files? [y/n] " -n 1 -r
  echo
  if [[ $REPLY =~ ^[Yy]$ ]]; then
    for f in "${PROFILES[@]}"; do
      echo "Updating $f"
      echo >> "$f"
      echo "# Sequre path (added by install script)" >> "$f"
      echo "$EXPORT_COMMAND" >> "$f"
    done
  else
    echo "Skipping."
  fi
fi

echo ""
echo "Sequre successfully installed at: $(pwd)"
echo "Open a new terminal session or update your PATH to use sequre"
