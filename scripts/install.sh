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

update_profile () {
  if ! grep -F -q "$EXPORT_COMMAND" "$1"; then
    read -p "Update PATH in $1? [y/n] " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
      echo "Updating $1"
      echo >> "$1"
      echo "# Sequre path (added by install script)" >> "$1"
      echo "$EXPORT_COMMAND" >> "$1"
    else
      echo "Skipping."
    fi
  else
    echo "PATH already updated in $1; skipping update."
  fi
}

if [[ "$SHELL" == *zsh ]]; then
  if [ -e ~/.zshenv ]; then
    update_profile ~/.zshenv
  elif [ -e ~/.zshrc ]; then
    update_profile ~/.zshrc
  else
    echo "Could not find zsh configuration file to update PATH"
  fi
elif [[ "$SHELL" == *bash ]]; then
  if [ -e ~/.bash_profile ]; then
    update_profile ~/.bash_profile
  elif [ -e ~/.bash_login ]; then
    update_profile ~/.bash_login
  elif [ -e ~/.profile ]; then
    update_profile ~/.profile
  else
    echo "Could not find bash configuration file to update PATH"
  fi
else
  echo "Don't know how to update configuration file for shell $SHELL"
fi

echo ""
echo "Sequre successfully installed at: $(pwd)"
echo "Open a new terminal session or update your PATH to use sequre"
