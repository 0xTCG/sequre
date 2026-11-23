#!/usr/bin/env bash
# Smoke test of an installed Sequre release (the CI runs it on every platform after the build; it also runs against
# any install prefix). Run it from the repository root: it uses examples/addmul.codon and scripts/.
#
#   scripts/smoke_test.sh <install prefix> [<Sequre archive>]
#
# Checks (each prints "ok" or "FAIL" and a reason; the exit status is 1 if any failed):
#   machine-types   every binary of Sequre's in the install (the launcher, the shim, the plugins), and in the archive
#                   if given, is built for this machine (scripts/check_machine_types.sh)
#   launcher-files  sock.* files in the working directory survive `sequre --help` and `sequre build`
#   run             `sequre run examples/addmul.codon --local`, with the MHE setup (the alignment shim and GMP), prints
#                   the results and leaves no socket file behind
#   build           `sequre build examples/addmul.codon`: the launcher runs the codon and the plugin of its own install
#                   (it finds itself), and on x86-64 Linux the shim comes before libcodonrt in the binary's NEEDED list
#   binary-N        the built binary, run directly in a clean environment (env -i: no LD_PRELOAD, SEQURE_GMP_PATH or
#                   CODON_PYTHON) with the MHE setup and N parties (3 and 4), prints the results, loads the GMP bundled
#                   with the plugin (Linux and macOS) and leaves no socket file behind
#   shim            x86-64 Linux: shrinking and freeing over-aligned blocks leaves their neighbours intact
#   seq             a program that uses the seq plugin and Sequre, built with `sequre build -release -plugin seq`, runs
#   installer-*     with the archive given, scripts/install.sh: a download that fails (HTTP 404), a broken archive and
#                   an archive whose launcher does not run each stop it with a non-zero status and leave an existing
#                   install (a scratch one) as it was; with no terminal and the script on stdin (curl | bash) it
#                   completes and leaves shell profiles alone, and SEQURE_UPDATE_PATH=yes adds one PATH line. Every
#                   installer run gets a scratch HOME and a scratch install directory.
# Environment: SMOKE_TIMEOUT (seconds per program run, default 600), SMOKE_INSTALL_SH (the installer to test,
# default scripts/install.sh).
set -uo pipefail

if [ $# -lt 1 ] || [ $# -gt 2 ]; then
  echo "usage: $0 <install prefix> [<Sequre archive>]" >&2
  exit 2
fi
REPO=$(cd "$(dirname "$0")/.." && pwd -P)
PREFIX=$(cd "$1" && pwd -P) || exit 2
ARCHIVE=
if [ $# -eq 2 ]; then
  ARCHIVE=$(cd "$(dirname "$2")" && pwd -P)/$(basename "$2")
fi
SEQURE=$PREFIX/bin/sequre
TIMEOUT=${SMOKE_TIMEOUT:-600}
INSTALL_SH=${SMOKE_INSTALL_SH:-$REPO/scripts/install.sh}
OS=$(uname -s | awk '{print tolower($0)}')
PLATFORM=$OS-$(uname -m)
case "$OS" in
  darwin) DLL=dylib ;;
  *) DLL=so ;;
esac
BUNDLED_GMP=$PREFIX/lib/codon/plugins/sequre/lib/libgmp.$DLL

WORK=$(mktemp -d "${TMPDIR:-/tmp}/sequre-smoke.XXXXXX")
WORK=$(cd "$WORK" && pwd -P)
trap 'rm -rf "$WORK"' EXIT
FAILED=0
ok() { echo "ok    $1${2:+: $2}"; }
fail() {
  echo "FAIL  $1: $2"
  FAILED=$((FAILED + 1))
}

# run_timeout <seconds> <command...>: runs the command in a process group of its own (a local run forks one process
# per party) and kills the group when the time is up. The watchdog has a group of its own too, so that killing it
# also ends its sleep.
run_timeout() {
  local limit=$1 pid dog rc
  shift
  set -m
  "$@" &
  pid=$!
  (
    sleep "$limit"
    kill -TERM -- -"$pid" 2> /dev/null
    sleep 5
    kill -KILL -- -"$pid" 2> /dev/null
  ) > /dev/null 2>&1 &
  dog=$!
  set +m
  wait "$pid"
  rc=$?
  kill -- -"$dog" 2> /dev/null
  wait "$dog" 2> /dev/null
  return $rc
}

# snapshot <dir>: a listing of every file's checksum and size, and every symlink's target, under dir (tar keeps an
# archive's modification times, so times cannot tell whether a file was replaced)
snapshot() {
  (
    cd "$1" || exit 1
    find . -type f -exec cksum {} + | sort -k3
    find . -type l | sort | while IFS= read -r l; do echo "$l -> $(readlink "$l")"; done
  )
}

# The socket files a local run left in directory $1 ("sock." and digits: the runtime's names)
run_sockets() { (cd "$1" && ls -1 | grep -E '^sock\.[0-9]+$' | tr '\n' ' '); }

# check_results <log> <parties>: every compute party printed both results of examples/addmul.codon
check_results() {
  local want=$(($2 - 1)) a b
  a=$(grep -c 'addmul: 471' "$1")
  b=$(grep -c 'innerprod: 32' "$1")
  [ "$a" -eq "$want" ] && [ "$b" -eq "$want" ]
}

echo "Smoke test of $PREFIX ($PLATFORM)"

# machine-types (Codon's own files come from its release and are not checked here)
targets=("$PREFIX/bin/sequre" "$PREFIX/lib/codon/plugins/sequre")
[ -d "$PREFIX/lib/codon/plugins/seq" ] && targets+=("$PREFIX/lib/codon/plugins/seq")
[ -f "$PREFIX/lib/sequre_align64.so" ] && targets+=("$PREFIX/lib/sequre_align64.so")
[ -n "$ARCHIVE" ] && targets+=("$ARCHIVE")
if out=$(sh "$REPO/scripts/check_machine_types.sh" "$PLATFORM" "${targets[@]}" 2>&1); then
  ok machine-types "$(echo "$out" | tail -1)"
else
  echo "$out" | grep -v '^ok' | sed 's/^/      /'
  fail machine-types "binaries built for another machine"
fi

# launcher-files
D=$WORK/launcher-files
mkdir -p "$D"
echo "a,b" > "$D/sock.csv"
echo stray > "$D/sock.9999"
(cd "$D" && "$SEQURE" --help > help.txt 2>&1)
(cd "$D" && run_timeout "$TIMEOUT" env CODON_DEBUG=0 "$SEQURE" build "$REPO/examples/addmul.codon" -o "$D/addmul" \
  > build.txt 2>&1)
if [ "$(cat "$D/sock.csv" 2> /dev/null)" = "a,b" ] && [ "$(cat "$D/sock.9999" 2> /dev/null)" = stray ]; then
  ok launcher-files "sock.csv and sock.9999 kept"
else
  fail launcher-files "sock.csv or sock.9999 deleted by \`sequre --help\` or \`sequre build\`"
fi

# run
D=$WORK/run
mkdir -p "$D"
echo "a,b" > "$D/sock.csv"
(cd "$D" && run_timeout "$TIMEOUT" env SEQURE_CP_COUNT=3 "$SEQURE" run "$REPO/examples/addmul.codon" --local \
  > log.txt 2>&1)
rc=$?
if [ $rc -ne 0 ] || ! check_results "$D/log.txt" 3; then
  tail -20 "$D/log.txt" | sed 's/^/      /'
  fail run "exit status $rc"
elif [ -n "$(run_sockets "$D")" ] || [ ! -f "$D/sock.csv" ]; then
  fail run "socket files left: $(run_sockets "$D") (sock.csv present: $([ -f "$D/sock.csv" ] && echo yes || echo no))"
else
  ok run "results from every compute party, no socket file left"
fi

# build
D=$WORK/build
mkdir -p "$D"
BIN=$D/addmul
(cd "$D" && run_timeout "$TIMEOUT" env CODON_DEBUG=0 SEQURE_TRACE_ARGS=1 "$SEQURE" build "$REPO/examples/addmul.codon" \
  -o "$BIN" > build.txt 2>&1)
rc=$?
if [ $rc -ne 0 ] || [ ! -x "$BIN" ]; then
  tail -20 "$D/build.txt" | sed 's/^/      /'
  fail build "exit status $rc"
  BIN=
elif ! grep -q -F "sequre: exec $PREFIX/bin/codon build " "$D/build.txt" ||
     ! grep -q -F -- "--plugin=$PREFIX/bin/../lib/codon/plugins/sequre " "$D/build.txt"; then
  fail build "the launcher did not run its own install's codon and plugin: $(grep 'sequre: exec' "$D/build.txt" | cut -c1-200)"
elif [ "$PLATFORM" = "linux-x86_64" ] && command -v readelf > /dev/null 2>&1; then
  needed=$(readelf -d "$BIN" | awk '/NEEDED/ { gsub(/[][]/, "", $NF); print $NF }' | tr '\n' ' ')
  case " $needed" in
    *" sequre_align64.so "*"libcodonrt.so"*) ok build "NEEDED: $needed" ;;
    *) fail build "the alignment shim is not ahead of libcodonrt.so in NEEDED: $needed" ;;
  esac
else
  ok build
fi

# binary-3, binary-4
if [ -n "$BIN" ]; then
  for n in 3 4; do
    D=$WORK/binary-$n
    mkdir -p "$D"
    trace=()
    case "$OS" in
      linux) trace=(LD_DEBUG=libs LD_DEBUG_OUTPUT="$D/ld") ;;
      darwin) trace=(DYLD_PRINT_LIBRARIES=1) ;;
    esac
    (cd "$D" && run_timeout "$TIMEOUT" env -i HOME="$HOME" PATH=/usr/bin:/bin SEQURE_CP_COUNT=$n ${trace[@]+"${trace[@]}"} \
      "$BIN" --local > log.txt 2>&1)
    rc=$?
    gmp=$(cat "$D"/ld.* "$D/log.txt" 2> /dev/null | grep -o -E '/[^ ]*libgmp[^ ]*' | sort -u | tr '\n' ' ')
    if [ $rc -ne 0 ] || ! check_results "$D/log.txt" $n; then
      grep -v -E '^ *[0-9]+:|^dyld' "$D/log.txt" | tail -20 | sed 's/^/      /'
      fail binary-$n "exit status $rc"
    elif [ -n "$(run_sockets "$D")" ]; then
      fail binary-$n "socket files left: $(run_sockets "$D")"
    elif [ "$gmp" != "$BUNDLED_GMP " ]; then
      fail binary-$n "loaded GMP [$gmp], not the bundled $BUNDLED_GMP"
    else
      ok binary-$n "clean environment, MHE setup, results from every compute party; GMP $BUNDLED_GMP"
    fi
  done
fi

# shim
if [ "$PLATFORM" = "linux-x86_64" ]; then
  D=$WORK/shim
  mkdir -p "$D"
  cat > "$D/shim.codon" << 'EOF'
import internal.gc as gc

N = 256
SIZE = 100
blocks = [Ptr[byte](gc.alloc(SIZE)) for _ in range(N)]
for b in blocks:
    for j in range(SIZE):
        b[j] = byte(0xAB)
for i in range(0, N, 4):
    blocks[i] = Ptr[byte](gc.realloc(blocks[i].as_byte(), 10, SIZE))  # shrink below the alignment
for i in range(2, N, 4):
    gc.free(blocks[i].as_byte())
extra = [Ptr[byte](gc.alloc(SIZE)) for _ in range(2 * N)]
for e in extra:
    for j in range(SIZE):
        e[j] = byte(0x11)
damaged = 0
for i in range(1, N, 2):
    for j in range(SIZE):
        if blocks[i][j] != byte(0xAB):
            damaged += 1
            break
print("damaged:", damaged)
EOF
  (cd "$D" && run_timeout "$TIMEOUT" env CODON_DEBUG=0 "$SEQURE" build -release shim.codon -o shim > build.txt 2>&1)
  # Preloaded as well as linked: the check is about the shim, whichever way a binary gets it
  out=$(cd "$D" && env -i PATH=/usr/bin:/bin LD_PRELOAD="$PREFIX/lib/sequre_align64.so" ./shim 2>&1)
  if [ "$out" = "damaged: 0" ]; then
    ok shim "no neighbour damaged by shrinks and frees"
  else
    fail shim "${out:-the program did not build or run: $(tail -3 "$D/build.txt")}"
  fi
fi

# seq: the seq plugin bundled with the release loads alongside Sequre's (the launcher puts -plugin seq first)
if [ -f "$PREFIX/lib/codon/plugins/seq/plugin.toml" ]; then
  D=$WORK/seq
  mkdir -p "$D"
  cat > "$D/seq.codon" << 'EOF'
from bio import *
from sequre import main, Sharetensor as Stensor

@main
def run(mpc, n):
    x = Stensor.enc(mpc, n)
    print(f"CP{mpc.pid}: {~s'ACGTTT'} {x.reveal(mpc)}")

run(len(s'ACGTACGT'))
EOF
  (cd "$D" && run_timeout "$TIMEOUT" env CODON_DEBUG=0 "$SEQURE" build -release -plugin seq seq.codon -o seqbin \
    > build.txt 2>&1)
  rc=$?
  if [ $rc -eq 0 ]; then
    (cd "$D" && run_timeout "$TIMEOUT" env -i HOME="$HOME" PATH=/usr/bin:/bin ./seqbin --local --skip-mhe-setup \
      > log.txt 2>&1)
  fi
  if [ $rc -ne 0 ]; then
    tail -5 "$D/build.txt" | sed 's/^/      /'
    fail seq "\`sequre build -release -plugin seq\` failed with exit status $rc"
  elif [ "$(grep -c 'AAACGT 8' "$D/log.txt")" -ne 2 ]; then
    tail -5 "$D/log.txt" | sed 's/^/      /'
    fail seq "the program did not print its results"
  else
    ok seq "bio and Sequre in one release build"
  fi
fi

# installer-*
if [ -n "$ARCHIVE" ]; then
  # A stand-in for Codon's archive (the checks are about the installer, not Codon), an archive whose launcher does not
  # run, and a scratch existing install the failing runs must leave alone
  mkdir -p "$WORK/fake/codon-fake/bin" "$WORK/fake/bad"
  printf '#!/bin/sh\n' > "$WORK/fake/codon-fake/bin/codon"
  chmod +x "$WORK/fake/codon-fake/bin/codon"
  FAKE_CODON=$WORK/fake/codon.tar.gz
  tar -czf "$FAKE_CODON" -C "$WORK/fake" codon-fake
  echo "<html><body>Not Found</body></html>" > "$WORK/fake/broken.tar.gz"
  tar -xzf "$ARCHIVE" -C "$WORK/fake/bad"
  printf '#!/bin/sh\nexit 1\n' > "$WORK/fake/bad/bin/sequre"
  tar -czf "$WORK/fake/bad-launcher.tar.gz" -C "$WORK/fake/bad" .
  MISSING_URL=https://github.com/0xTCG/sequre/releases/download/v0.0.0/sequre-smoke-test-missing.tar.gz
  EXISTING=$WORK/existing
  mkdir -p "$EXISTING/bin" "$EXISTING/lib/codon/plugins/sequre"
  cp "$SEQURE" "$EXISTING/bin/sequre"
  printf '#!/bin/sh\n# the existing install'"'"'s codon\n' > "$EXISTING/bin/codon"
  echo "existing" > "$EXISTING/lib/codon/plugins/sequre/plugin.toml"
  BEFORE=$(snapshot "$EXISTING")

  # Detaches a command from the controlling terminal, so that the installer finds none to ask on. A process group
  # leader (run_timeout's job) cannot start a session: setsid forks first, and so does the perl fallback (macOS).
  detach=()
  if command -v setsid > /dev/null 2>&1; then
    detach=(setsid -w)
  elif command -v perl > /dev/null 2>&1; then
    detach=(perl -MPOSIX -e 'my $p = fork() // die "fork: $!"; if ($p) { waitpid($p, 0); exit($? & 127 ? 1 : $? >> 8) }
      POSIX::setsid() or die "setsid: $!"; exec @ARGV or die "exec: $!"')
  fi

  # installer_fails <check> <sequre archive>: the installer stops, says so, and leaves the existing install as it was
  installer_fails() {
    local check=$1 src=$2 h=$WORK/$1 after
    mkdir -p "$h"
    run_timeout "$TIMEOUT" env HOME="$h" SEQURE_INSTALL_DIR="$EXISTING" SEQURE_CODON_ARCHIVE="$FAKE_CODON" \
      SEQURE_ARCHIVE="$src" SEQURE_UPDATE_PATH=no bash "$INSTALL_SH" > "$h/log.txt" 2>&1 < /dev/null
    rc=$?
    after=$(snapshot "$EXISTING")
    if [ $rc -eq 0 ]; then
      fail "$check" "exit status 0"
    elif [ "$after" != "$BEFORE" ]; then
      fail "$check" "the existing install changed"
      diff <(echo "$BEFORE") <(echo "$after") | head -6 | sed 's/^/      /'
    elif ! grep -q -i 'error' "$h/log.txt"; then
      fail "$check" "no error message"
    else
      ok "$check" "exit status $rc, existing install unchanged: $(grep -m1 '^error' "$h/log.txt" | cut -c1-150)"
    fi
  }
  installer_fails installer-404 "$MISSING_URL"
  installer_fails installer-broken "$WORK/fake/broken.tar.gz"
  installer_fails installer-bad-launcher "$WORK/fake/bad-launcher.tar.gz"

  # installer-no-tty: the script on stdin (curl | bash), detached from any terminal
  h=$WORK/installer-no-tty
  mkdir -p "$h"
  echo "# profile" > "$h/.bashrc"
  cat "$INSTALL_SH" | run_timeout "$TIMEOUT" env HOME="$h" SEQURE_INSTALL_DIR="$h/prefix" \
    SEQURE_CODON_ARCHIVE="$FAKE_CODON" SEQURE_ARCHIVE="$ARCHIVE" ${detach[@]+"${detach[@]}"} bash > "$h/log.txt" 2>&1
  rc=$?
  if [ $rc -ne 0 ] || [ ! -x "$h/prefix/bin/sequre" ]; then
    tail -5 "$h/log.txt" | sed 's/^/      /'
    fail installer-no-tty "exit status $rc"
  elif [ "$(cat "$h/.bashrc")" != "# profile" ]; then
    fail installer-no-tty ".bashrc changed without being asked"
  elif ! grep -q 'No terminal to ask on' "$h/log.txt"; then
    fail installer-no-tty "no message about the skipped PATH update"
  else
    ok installer-no-tty "installed, profile untouched, PATH update skipped with a message"
  fi

  # installer-update-path: SEQURE_UPDATE_PATH=yes answers the question
  h=$WORK/installer-update-path
  mkdir -p "$h"
  echo "# profile" > "$h/.bashrc"
  run_timeout "$TIMEOUT" env HOME="$h" SEQURE_INSTALL_DIR="$h/prefix" SEQURE_CODON_ARCHIVE="$FAKE_CODON" \
    SEQURE_ARCHIVE="$ARCHIVE" SEQURE_UPDATE_PATH=yes bash "$INSTALL_SH" > "$h/log.txt" 2>&1 < /dev/null
  rc=$?
  lines=$(grep -c -F "export PATH=$h/prefix/bin:\$PATH" "$h/.bashrc")
  if [ $rc -eq 0 ] && [ "$lines" -eq 1 ]; then
    ok installer-update-path "one PATH line added"
  else
    fail installer-update-path "exit status $rc, $lines PATH lines in .bashrc"
  fi
fi

if [ "$FAILED" -gt 0 ]; then
  echo "$FAILED check(s) failed"
  exit 1
fi
echo "all checks passed"
