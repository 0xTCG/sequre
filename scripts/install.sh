#!/usr/bin/env bash
# Installs Sequre, with the Codon release it runs on, from the release archives for this platform.
#
#   /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/0xTCG/sequre/develop/scripts/install.sh)"
#   curl -fsSL https://raw.githubusercontent.com/0xTCG/sequre/develop/scripts/install.sh | bash
#
# Both archives are downloaded into a temporary directory and checked, and the new launcher must run on this machine,
# before either is unpacked: a missing asset, a failed download or an archive for another platform stops the install
# with a message naming it, and leaves an existing install as it was. Then the directories each archive owns are
# replaced, each named in the output: Codon's lib/codon/stdlib and include/codon, and Sequre's lib/codon/plugins/sequre
# (moved to lib/codon/plugins/sequre.previous when it holds entries the new archive lacks, a user's files say, but not
# for a lib/ of nothing but the shared libraries that earlier releases bundled there); the seq
# plugin that Sequre's archives for Codon 0.19 bundled goes too, as it does not load into Codon 0.20. Codon and Sequre
# are then unpacked over the install directory (other files an earlier install has and the new one lacks stay); a
# failure there (permissions, disk space) says that the install is incomplete. An install directory whose lib,
# lib/codon, lib/codon/plugins or include is a symbolic link is refused before anything is downloaded: replacing the
# directories under it would change the link's target, outside the install.
#
# Environment (all optional):
#   SEQURE_INSTALL_DIR    install directory (default: ~/.sequre; a relative path is taken from the current one)
#   SEQURE_ARCHIVE        the Sequre archive, a URL or a local file (default: the latest release's archive for this
#                         platform, https://github.com/0xTCG/sequre/releases/latest/download/sequre-<os>-<arch>.tar.gz)
#   SEQURE_CODON_ARCHIVE  the Codon archive, a URL or a local file (default: Codon's release archive for this platform
#                         and the Codon version the Sequre archive was built for, which its
#                         lib/codon/plugins/sequre/codon-version names, v0.19.6 when it has none (v0.1.6 and earlier):
#                         https://github.com/exaloop/codon/releases/download/<version>/codon-<os>-<arch>.tar.gz)
#   SEQURE_UPDATE_PATH    yes or no: whether to add the install's bin directory to PATH in your shell profiles.
#                         Unset, the script asks on the terminal; with no terminal (CI, a pipe without a tty) it skips
#                         the update and says so.
set -euo pipefail

SEQURE_INSTALL_DIR=${SEQURE_INSTALL_DIR:-$HOME/.sequre}
case "$SEQURE_INSTALL_DIR" in
  /*) ;;
  *) SEQURE_INSTALL_DIR=$PWD/$SEQURE_INSTALL_DIR ;;
esac
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

# The directories the install replaces lie under these; through a symbolic link, replacing them would change the
# link's target, outside the install
for d in lib lib/codon lib/codon/plugins include; do
  if [ -L "$SEQURE_INSTALL_DIR/$d" ]; then
    echo "error: $SEQURE_INSTALL_DIR/$d is a symbolic link (to $(readlink "$SEQURE_INSTALL_DIR/$d"))," >&2
    echo "and the installer replaces Codon's and Sequre's directories under it, which would change the" >&2
    echo "link's target. Install into another directory (SEQURE_INSTALL_DIR), or replace the link with a" >&2
    echo "copy of what it points to." >&2
    echo "Nothing was installed." >&2
    exit 1
  fi
done

CODON_BUILD_ARCHIVE=codon-$OS-$ARCH.tar.gz
SEQURE_BUILD_ARCHIVE=sequre-$OS-$ARCH.tar.gz
SEQURE_RELEASES=https://github.com/0xTCG/sequre/releases
SEQURE_SRC=${SEQURE_ARCHIVE:-$SEQURE_RELEASES/latest/download/$SEQURE_BUILD_ARCHIVE}

TMP=$(mktemp -d "${TMPDIR:-/tmp}/sequre-install.XXXXXX")
trap 'rm -rf "$TMP"' EXIT

if [ -t 2 ]; then
  CURL_PROGRESS=(--progress-bar)
else
  CURL_PROGRESS=(--silent --show-error)
fi

# fetch <what> <URL or file> <destination>: downloads (curl -f: an HTTP error is a failure, not a page saved as the
# archive) or copies a local file.
fetch() {
  local what=$1 src=$2 dest=$3
  case "$src" in
    http://* | https://* | file://*)
      echo "Downloading $what: $src"
      if ! curl -fL --retry 3 --retry-delay 2 "${CURL_PROGRESS[@]}" -o "$dest" "$src"; then
        echo "error: could not download $what from $src" >&2
        return 1
      fi
      ;;
    *)
      if [ ! -f "$src" ]; then
        echo "error: $what: no such file: $src" >&2
        return 1
      fi
      echo "Using $what: $src"
      cp "$src" "$dest"
      ;;
  esac
}

# check_archive <what> <file> [<member> <member regex>]...: the file is a complete gzip-compressed tar archive that
# holds these members.
check_archive() {
  local what=$1 file=$2
  shift 2
  if ! tar -tzf "$file" > "$file.list" 2> "$file.err"; then
    echo "error: $what is not a complete gzip-compressed tar archive: $(head -c 300 "$file.err")" >&2
    return 1
  fi
  while [ $# -ge 2 ]; do
    if ! grep -q -E "$2" "$file.list"; then
      echo "error: $what does not hold $1" >&2
      return 1
    fi
    shift 2
  done
}

echo "Installing Sequre to $SEQURE_INSTALL_DIR ..."

# Sequre's archive first: it is small, and the one a release may lack
if ! fetch "the Sequre archive ($SEQURE_BUILD_ARCHIVE)" "$SEQURE_SRC" "$TMP/sequre.tar.gz"; then
  if [ -z "${SEQURE_ARCHIVE:-}" ]; then
    echo "The latest release may have no archive for $OS-$ARCH: see $SEQURE_RELEASES." >&2
  fi
  echo "Nothing was installed." >&2
  exit 1
fi
if ! check_archive "the Sequre archive ($SEQURE_SRC)" "$TMP/sequre.tar.gz" bin/sequre '^(\./)?bin/sequre$' \
     lib/codon/plugins/sequre/plugin.toml '^(\./)?lib/codon/plugins/sequre/plugin\.toml$'; then
  echo "Nothing was installed." >&2
  exit 1
fi

# The Codon release the Sequre archive was built for: the version its plugin's codon-version file names. An archive
# without one (v0.1.6 and earlier) was built for v0.19.6.
CODON_VERSION=v0.19.6
VERSION_MEMBER=$(grep -m 1 -E '^(\./)?lib/codon/plugins/sequre/codon-version$' "$TMP/sequre.tar.gz.list" || true)
if [ -n "$VERSION_MEMBER" ]; then
  if ! CODON_VERSION=$(tar -xzOf "$TMP/sequre.tar.gz" "$VERSION_MEMBER" | tr -d '[:space:]') ||
     ! echo "$CODON_VERSION" | grep -q -E '^v[0-9]+\.[0-9]+\.[0-9]+$'; then
    echo "error: the Sequre archive ($SEQURE_SRC) names no valid Codon version in $VERSION_MEMBER: '$CODON_VERSION'" >&2
    echo "Nothing was installed." >&2
    exit 1
  fi
fi
CODON_SRC=${SEQURE_CODON_ARCHIVE:-https://github.com/exaloop/codon/releases/download/$CODON_VERSION/$CODON_BUILD_ARCHIVE}

if ! fetch "the Codon $CODON_VERSION archive ($CODON_BUILD_ARCHIVE)" "$CODON_SRC" "$TMP/codon.tar.gz"; then
  echo "Nothing was installed." >&2
  exit 1
fi
if ! check_archive "the Codon archive ($CODON_SRC)" "$TMP/codon.tar.gz" bin/codon '^[^/]+/bin/codon$'; then
  echo "Nothing was installed." >&2
  exit 1
fi

# The new launcher must run on this machine before anything is unpacked: an archive for another platform, or one
# built for a newer OS, stops the install here. Run from the temporary directory: older launchers delete sock.* files
# in the directory they run in.
LAUNCHER=$(grep -m 1 -E '^(\./)?bin/sequre$' "$TMP/sequre.tar.gz.list")
mkdir -p "$TMP/probe"
if ! tar -xzf "$TMP/sequre.tar.gz" -C "$TMP/probe" "$LAUNCHER" ||
   ! (cd "$TMP/probe" && "./$LAUNCHER" --help > /dev/null 2>&1); then
  echo "error: the launcher in the Sequre archive ($SEQURE_SRC) does not run on this machine ($OS-$ARCH)" >&2
  echo "Nothing was installed." >&2
  exit 1
fi

# 1. Codon, then 2. Sequre (plugin, launcher, alignment shim) on top. Each archive owns some directories outright, so
# they are replaced, not unpacked over: an install over an earlier one would keep that one's files that this one lacks
# (Codon's stdlib and headers of another release, the files an older plugin had).
in_sequre_archive() {  # in_sequre_archive <path>: the Sequre archive has <path> or something under it
  awk -v p="$1" '{ sub(/^\.\//, ""); if ($0 == p || $0 == p "/" || index($0, p "/") == 1) { found = 1; exit } }
                 END { exit !found }' "$TMP/sequre.tar.gz.list"
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
clear_owned_directories() {
  plugins="$SEQURE_INSTALL_DIR/lib/codon/plugins"
  if [ -e "$SEQURE_INSTALL_DIR/lib/codon/stdlib" ] || [ -e "$SEQURE_INSTALL_DIR/include/codon" ]; then
    echo "Replacing Codon's stdlib and headers (lib/codon/stdlib, include/codon)."
    rm -rf "$SEQURE_INSTALL_DIR/lib/codon/stdlib" "$SEQURE_INSTALL_DIR/include/codon" || return 1
  fi
  # The seq plugin that Sequre's archives for Codon 0.19 bundled (their plugin has no codon-version) does not load into
  # Codon 0.20; a seq installed beside a later Sequre is the user's own and stays
  if [ -d "$plugins/seq" ] && [ -d "$plugins/sequre" ] && [ ! -f "$plugins/sequre/codon-version" ] &&
     ! in_sequre_archive lib/codon/plugins/seq; then
    echo "Removing the seq plugin that the earlier Sequre release bundled (built for Codon 0.19, it does not load into"
    echo "Codon $CODON_VERSION)."
    rm -rf "$plugins/seq" || return 1
  fi
  # Sequre's plugin directory comes whole from the archive. Entries the archive lacks (a backup of the user's, an
  # earlier layout's files) are kept: the old directory is moved aside instead of removed.
  if [ -e "$plugins/sequre" ] || [ -L "$plugins/sequre" ]; then
    extra=""
    for entry in "$plugins/sequre"/* "$plugins/sequre"/.[!.]* "$plugins/sequre"/..?*; do
      if [ -e "$entry" ] || [ -L "$entry" ]; then
        # lib: the shared libraries that earlier releases bundled beside the plugin, which this archive does not: the
        # archive's own, not the user's, as long as it holds nothing else
        if [ "${entry##*/}" = lib ] && ! in_sequre_archive lib/codon/plugins/sequre/lib && only_shared_libraries "$entry"; then
          continue
        fi
        in_sequre_archive "lib/codon/plugins/sequre/${entry##*/}" || extra="$extra ${entry##*/}"
      fi
    done
    if [ -n "$extra" ]; then
      rm -rf "$plugins/sequre.previous" || return 1
      mv "$plugins/sequre" "$plugins/sequre.previous" || return 1
      echo "Moved the earlier Sequre plugin to $plugins/sequre.previous: it holds entries this release lacks"
      echo "(${extra# }); delete it when you no longer need them."
    else
      echo "Replacing the Sequre plugin (lib/codon/plugins/sequre)."
      rm -rf "$plugins/sequre" || return 1
    fi
  fi
}
if ! mkdir -p "$SEQURE_INSTALL_DIR" ||
   ! clear_owned_directories ||
   ! tar -xzf "$TMP/codon.tar.gz" -C "$SEQURE_INSTALL_DIR" --strip-components=1 ||
   ! tar -xzf "$TMP/sequre.tar.gz" -C "$SEQURE_INSTALL_DIR"; then
  echo "error: unpacking into $SEQURE_INSTALL_DIR failed (see above), so the install there is incomplete." >&2
  echo "Fix the cause (permissions, disk space) and rerun the installer." >&2
  exit 1
fi

# 3. Codon 0.19 only (the archives of Sequre v0.1.6 and earlier), Linux x86_64: Codon emits aligned vector moves for
#    heap vectors, which the collector only 16-byte-aligns, and 0.19.6's Ptr load/store are patched to be unaligned
#    (those releases' `sequre build` binaries run without the over-align shim). Codon 0.20 needs no patch: its own
#    simd.codon makes loads and stores through Ptr[Vec] unaligned, and the over-align shim, which the launcher preloads
#    for `sequre run` and links into the binaries of `sequre build`, covers the other aligned moves (tuples of vectors,
#    struct fields, generator frames). Inlined (not delegated to a sibling script) so that it also works when this
#    installer is run via `curl ... | bash`.
if [ "$OS" = "linux" ] && [ "$ARCH" = "x86_64" ] && [ "${CODON_VERSION#v0.19.}" != "$CODON_VERSION" ]; then
  PTR="$SEQURE_INSTALL_DIR/lib/codon/stdlib/internal/types/ptr.codon"
  if [ -f "$PTR" ] && ! grep -q ', align 1' "$PTR"; then
    sed -i -E '/(^|[[:space:]])(load|store)[[:space:]]/ { /align/! { /%0$/ s/$/, align 1/ } }' "$PTR"
    echo "Applied the x86-64 Ptr unaligned-load patch (Codon $CODON_VERSION)."
  fi
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
  REPLY=${SEQURE_UPDATE_PATH:-}
  if [ -z "$REPLY" ]; then
    # Read the answer from the terminal, never from a stdin that is not one: under `curl ... | bash` stdin is the
    # script itself.
    if [ -t 0 ]; then
      read -p "Update PATH in the above files? [y/n] " -n 1 -r || REPLY=
      echo
    elif { : < /dev/tty; } 2> /dev/null; then
      read -p "Update PATH in the above files? [y/n] " -n 1 -r < /dev/tty || REPLY=
      echo
    else
      echo "No terminal to ask on, so PATH was not updated. Add the export command above to your shell profile,"
      echo "or rerun with SEQURE_UPDATE_PATH=yes."
      REPLY=skip
    fi
  fi
  case "$REPLY" in
    [Yy] | [Yy][Ee][Ss])
      for f in "${PROFILES[@]}"; do
        echo "Updating $f"
        echo >> "$f"
        echo "# Sequre path (added by install script)" >> "$f"
        echo "$EXPORT_COMMAND" >> "$f"
      done
      ;;
    skip) ;;
    *) echo "Skipping." ;;
  esac
fi

echo ""
echo "Sequre successfully installed at: $SEQURE_INSTALL_DIR"
echo "Open a new terminal session or update your PATH to use sequre"
