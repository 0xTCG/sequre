#!/usr/bin/env bash
set -e
set -o pipefail

SEQURE_INSTALL_DIR=~/.sequre
OS=$(uname -s | awk '{print tolower($0)}')
ARCH=$(uname -m)

if [ "$OS" != "linux" ]; then
  echo "error: Pre-built binaries only exist for Linux (x86_64, aarch64)." >&2
  exit 1
fi

if [ "$ARCH" != "x86_64" ] && [ "$ARCH" != "aarch64" ]; then
  echo "error: Pre-built binaries only exist for x86_64 and aarch64." >&2
  exit 1
fi

CODON_VERSION=v0.17.0
CODON_BUILD_ARCHIVE=codon-$OS-$ARCH.tar.gz
SEQURE_BUILD_ARCHIVE=sequre-$OS-$ARCH.tar.gz

echo "Installing Sequre to $SEQURE_INSTALL_DIR ..."

mkdir -p "$SEQURE_INSTALL_DIR"
cd "$SEQURE_INSTALL_DIR"

# 1. Install Codon runtime
echo "Downloading Codon $CODON_VERSION ..."
curl -L "https://github.com/exaloop/codon/releases/download/$CODON_VERSION/$CODON_BUILD_ARCHIVE" | tar zxvf - --strip-components=1

# 2. Install Sequre (plugin + launcher) on top
echo "Downloading Sequre ..."
curl -L "https://github.com/0xTCG/sequre/releases/latest/download/$SEQURE_BUILD_ARCHIVE" | tar zxvf - --strip-components=0

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
