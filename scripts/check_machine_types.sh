#!/bin/sh
# Checks that every binary under the given paths is built for the given platform, so that a release archive never
# ships a binary of another architecture (an x86-64 libgmp in an aarch64 archive, say). Binaries are recognised by
# their magic number: ELF, Mach-O (thin or universal) and static archives (by their first object file).
#
#   scripts/check_machine_types.sh <platform> <file | directory | archive.tar.gz>...
#
# <platform> is linux-x86_64, linux-aarch64 or darwin-arm64 (the release archives' names). Prints one line per binary
# and exits 1 if any is built for another machine, 2 on a usage error. POSIX sh, od and dd only, so that it runs in
# the manylinux build images and on macOS.
set -eu

usage() {
  echo "usage: $0 <linux-x86_64|linux-aarch64|darwin-arm64> <file|directory|archive.tar.gz>..." >&2
  exit 2
}
[ $# -ge 2 ] || usage
target=$1
shift
case "$target" in
  linux-x86_64) want="elf 62" ;;          # EM_X86_64
  linux-aarch64) want="elf 183" ;;        # EM_AARCH64
  darwin-arm64) want="macho 16777228" ;;  # CPU_TYPE_ARM64
  *) usage ;;
esac

# The bytes of file $1 at offset $2, count $3, in hex ("7f 45 4c 46")
bytes() { od -An -v -t x1 -j "$2" -N "$3" "$1" 2> /dev/null | tr -s ' \n' '  ' | sed 's/^ //; s/ $//'; }
le16() { set -- $(bytes "$1" "$2" 2); [ $# -eq 2 ] && echo $((0x$2$1)); }
le32() { set -- $(bytes "$1" "$2" 4); [ $# -eq 4 ] && echo $((0x$4$3$2$1)); }
be32() { set -- $(bytes "$1" "$2" 4); [ $# -eq 4 ] && echo $((0x$1$2$3$4)); }

name_of() {
  case "$1 $2" in
    "elf 62") echo x86-64 ;;
    "elf 183") echo aarch64 ;;
    "elf 3") echo i386 ;;
    "elf 40") echo arm ;;
    "macho 16777228") echo arm64 ;;
    "macho 16777223") echo x86_64 ;;
    *) echo "$1 machine $2" ;;
  esac
}

# "<format> <machine>" of the object at offset $2 of file $1, or nothing for other data (a file too short to hold
# its machine field included)
object_at() {
  case "$(bytes "$1" "$2" 4)" in
    "7f 45 4c 46") m=$(le16 "$1" $(($2 + 18))) && echo "elf $m" ;;
    "cf fa ed fe" | "ce fa ed fe") m=$(le32 "$1" $(($2 + 4))) && echo "macho $m" ;;
  esac
  return 0
}

# The machines of file $1, one "<format> <machine>" per line (a universal binary has one per slice), or nothing for a
# file that is not a binary
machines() {
  f=$1
  case "$(bytes "$f" 0 4)" in
    "ca fe ba be")  # universal Mach-O: nfat_arch, then 20-byte entries starting with the cputype. A Java class
      # file has the same magic, followed by its version (45 and up where nfat_arch would be): not a binary here.
      n=$(be32 "$f" 4) || n=0
      if [ "$n" -ge 1 ] && [ "$n" -le 20 ]; then
        i=0
        while [ "$i" -lt "$n" ]; do
          echo "macho $(be32 "$f" $((8 + 20 * i)))"
          i=$((i + 1))
        done
      fi
      ;;
    "21 3c 61 72")  # "!<arch>": a static archive, judged by its first object file
      off=8
      size=$(dd if="$f" bs=1 skip=$((off + 48)) count=10 2> /dev/null | tr -d ' ')
      while [ -n "$size" ]; do
        data=$((off + 60))
        name=$(dd if="$f" bs=1 skip="$off" count=16 2> /dev/null | tr -d ' ')
        case "$name" in
          "#1/"*) start=$((data + ${name#\#1/})) ;;  # BSD long name: stored at the start of the data
          *) start=$data ;;
        esac
        obj=$(object_at "$f" "$start")
        if [ -n "$obj" ]; then
          echo "$obj"
          return 0
        fi
        off=$((data + size + size % 2))
        size=$(dd if="$f" bs=1 skip=$((off + 48)) count=10 2> /dev/null | tr -d ' ')
      done
      ;;
    *) object_at "$f" 0 ;;
  esac
}

tmp=$(mktemp -d "${TMPDIR:-/tmp}/check-machine-types.XXXXXX")
trap 'rm -rf "$tmp"' EXIT
: > "$tmp/files"
for p in "$@"; do
  if [ -d "$p" ]; then
    find "$p" -type f >> "$tmp/files"
  elif [ -f "$p" ]; then
    case "$p" in
      *.tar.gz | *.tgz)
        d="$tmp/archive$(wc -l < "$tmp/files" | tr -d ' ')"
        mkdir -p "$d"
        tar -xzf "$p" -C "$d"
        find "$d" -type f >> "$tmp/files"
        ;;
      *) echo "$p" >> "$tmp/files" ;;
    esac
  else
    echo "error: no such file or directory: $p" >&2
    exit 2
  fi
done

checked=0
wrong=0
while IFS= read -r f; do
  ms=$(machines "$f")
  [ -n "$ms" ] || continue
  checked=$((checked + 1))
  label=${f#"$tmp"/archive*/}
  ok=0
  names=""
  while IFS= read -r m; do
    [ "$m" = "$want" ] && ok=1
    names="$names $(name_of $m)"
  done << EOF
$ms
EOF
  if [ "$ok" -eq 1 ]; then
    echo "ok     $label:$names"
  else
    echo "WRONG  $label:$names (want $(name_of $want))"
    wrong=$((wrong + 1))
  fi
done < "$tmp/files"

if [ "$checked" -eq 0 ]; then
  echo "error: no binaries found in $*" >&2
  exit 1
fi
if [ "$wrong" -gt 0 ]; then
  echo "$wrong of $checked binaries are not built for $target" >&2
  exit 1
fi
echo "all $checked binaries are built for $target"
