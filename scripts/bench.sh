#!/usr/bin/env bash
# Sequre performance-regression benchmark driver (tests/bench/bench_{mpc,mhe,btp}.codon).
#
#   scripts/bench.sh run  [--label NAME] [--cases P1,P2] [--skip P1,P2] [--reps-scale F]
#                         [--binaries mpc,mhe,btp] [--cpus auto|LIST|none] [--pin auto|C0:C1:C2|none]
#                         [--timeout SEC] [--iso-timeout SEC]
#                         [--gc-adaptive | --gc-none] [--gc-heap 4G|none] [--rebuild] [--dry-run]
#   scripts/bench.sh list [--cases P1,P2] [--skip P1,P2]
#   scripts/bench.sh compare BASE_DIR CAND_DIR [bench_compare.py options]
#   scripts/bench.sh calibrate RUN1_DIR RUN2_DIR [--out FILE]
#
# run: builds the needed binaries in release mode (cached by a hash of tests/bench, the applications the bench files
# import, the whole stdlib and the toolchain; --rebuild forces a build), and runs each binary from a fresh temporary
# directory (local mode leaves sock.* files behind) under `timeout`, with the AVX alignment shim preloaded as the
# `sequre` launcher does. Cases flagged "iso" in the registry, and known-broken cases asked for by exact name
# (--cases name$), run in processes of their own after the rest.
# CPU placement is fixed so that runs do not differ by where the scheduler put the parties. --cpus auto (the
# default) restricts the binaries to one hardware thread per physical core; --pin auto splits those cores into
# parties + 1 equal blocks of consecutive cores and gives one to each party (CP0, CP1, CP2; bench_btp uses CP1's),
# leaving the last block to the system. On a 24-core machine that numbers its SMT siblings 24-47 this is
# --cpus 0-23 --pin 0-5:6-11:12-17. A party pins all of its threads, the collector's marker threads included.
# Both options also take explicit lists, or none; the manifest records the resolved values, and compare warns
# when two runs were placed (or configured) differently.
# Garbage collection: bench.sh passes --gc-each-rep, so every repetition starts right after an untimed full
# collection (the samples of allocation-heavy cases otherwise depend on where the heap is in its growth/collection
# cycle; measured: median run-to-run noise 2.3% with it, 3.5% without, +40 s per suite). --gc-adaptive leaves the
# binaries' default (collect at the start of every case and before repetitions that follow one longer than 0.25 s);
# --gc-none never forces a collection. Each party's file records whether its collector marked in parallel
# (gc_parallel; the runtime restarts the marker threads in forked parties, without which collections there run on one
# thread, ~10x slower). --gc-heap (default 4G) sets the collector's
# initial heap (GC_INITIAL_HEAP_SIZE, reserved lazily): the collection trigger scales with the heap size, so without
# it a case collects more often when run alone (small fresh heap) than late in the full suite -- e.g.
# mpc.beaver.mul_vec_1e4 measured 4.8 ms in the suite and 8.4 ms alone. Compare runs with the same --cases anyway.
# Every run is checked for completeness (each party's file must end with a __done__ row and hold every selected
# case), not by exit status. Results: results/bench/<label>/
#     manifest.json   git sha + dirty-tree hash, machine, load, shim, thread and HE settings, build and run times
#     summary.tsv     one row per case (the slowest compute party's median and min, MAD, witness, openings per
#                     repetition from harness v4 on)
#     runs/<binary>.<main|case>/CP<pid>.tsv, stdout.log, check.json      raw per-party results
#     build/<binary>.log
# Exit status: 0 complete, 1 a run incomplete (results kept), 2 usage.
#
# Targeted before/after example (case names select by prefix; `list` shows them):
#   scripts/bench.sh run --label before-fix --cases mpc.beaver,mpc.share
#   ... apply the fix ...
#   scripts/bench.sh run --label after-fix --cases mpc.beaver,mpc.share
#   scripts/bench.sh compare results/bench/before-fix results/bench/after-fix
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TOOL=(python3 "$REPO/scripts/bench_tool.py")
COMPARE=(python3 "$REPO/scripts/bench_compare.py")
SHIM="$HOME/.sequre/lib/sequre_align64.so"
GMP="$HOME/.sequre/lib/codon/plugins/sequre/lib/libgmp.so"
CACHE="$REPO/results/bench/.cache"

usage() { awk 'NR > 1 && /^#/ { sub(/^# ?/, ""); print; next } NR > 1 { exit }' "${BASH_SOURCE[0]}"; }

cmd="${1:-help}"
[[ $# -gt 0 ]] && shift
case "$cmd" in
    list) exec "${TOOL[@]}" list "$@" ;;
    compare) exec "${COMPARE[@]}" "$@" ;;
    calibrate) exec "${COMPARE[@]}" --calibrate "$@" ;;
    run) ;;
    help|-h|--help) usage; exit 0 ;;
    *) echo "bench.sh: unknown command '$cmd'" >&2; usage >&2; exit 2 ;;
esac

LABEL="" CASES="" SKIP="" REPS_SCALE="1" BINS="mpc,mhe,btp" CPUS="auto" TIMEOUT=1200 ISO_TIMEOUT=600 REBUILD=0 DRY=0
GC_MODE="each-rep" GC_HEAP="4G"
PIN="auto"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --label) LABEL="$2"; shift 2 ;;      --label=*) LABEL="${1#*=}"; shift ;;
        --cases) CASES="$2"; shift 2 ;;      --cases=*) CASES="${1#*=}"; shift ;;
        --skip) SKIP="$2"; shift 2 ;;        --skip=*) SKIP="${1#*=}"; shift ;;
        --reps-scale) REPS_SCALE="$2"; shift 2 ;;  --reps-scale=*) REPS_SCALE="${1#*=}"; shift ;;
        --binaries) BINS="$2"; shift 2 ;;    --binaries=*) BINS="${1#*=}"; shift ;;
        --cpus) CPUS="$2"; shift 2 ;;        --cpus=*) CPUS="${1#*=}"; shift ;;
        --pin) PIN="$2"; shift 2 ;;          --pin=*) PIN="${1#*=}"; shift ;;
        --timeout) TIMEOUT="$2"; shift 2 ;;  --timeout=*) TIMEOUT="${1#*=}"; shift ;;
        --iso-timeout) ISO_TIMEOUT="$2"; shift 2 ;;  --iso-timeout=*) ISO_TIMEOUT="${1#*=}"; shift ;;
        --rebuild) REBUILD=1; shift ;;
        --gc-each-rep) GC_MODE="each-rep"; shift ;;  # the default
        --gc-adaptive) GC_MODE="adaptive"; shift ;;
        --gc-none) GC_MODE="none"; shift ;;
        --gc-heap) GC_HEAP="$2"; shift 2 ;;  --gc-heap=*) GC_HEAP="${1#*=}"; shift ;;
        --dry-run) DRY=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "bench.sh: unknown option '$1'" >&2; exit 2 ;;
    esac
done
if [[ "$CPUS" != "none" ]] && ! command -v taskset >/dev/null 2>&1; then
    echo "bench.sh: taskset not found; not restricting the binaries' CPUs" >&2
    CPUS="none"
fi
LAYOUT="$("${TOOL[@]}" cpus --cpus "$CPUS" --pin "$PIN")" || exit 2
read -r CPUS PIN <<< "$LAYOUT"
[[ "$CPUS" == "-" ]] && CPUS=""
echo "== cpus: ${CPUS:-not restricted}; party pinning: $PIN"
GC_FLAGS=()
[[ "$GC_MODE" == "each-rep" ]] && GC_FLAGS+=(--gc-each-rep)
[[ "$GC_MODE" == "none" ]] && GC_FLAGS+=(--gc-none)

PLAN="$("${TOOL[@]}" plan --cases "$CASES" --skip "$SKIP" --binaries "$BINS")" || exit 2
echo "== plan"
echo "$PLAN" | awk '{ if ($2 == "main") { n = split($3, a, ","); print "  " $1 " main: " n " cases" } else print "  " $0 }'
NEEDED=()
for b in ${BINS//,/ }; do
    if echo "$PLAN" | grep -qE "^$b (main|iso) "; then NEEDED+=("$b"); fi
done
if [[ ${#NEEDED[@]} -eq 0 ]]; then echo "bench.sh: nothing selected" >&2; exit 2; fi
[[ $DRY -eq 1 ]] && exit 0

SHA="$(git -C "$REPO" rev-parse --short HEAD)"
DIRTY="$(git -C "$REPO" status --porcelain | grep -q . && echo "-dirty" || true)"
[[ -z "$LABEL" ]] && LABEL="$(date +%Y%m%d-%H%M%S)-$SHA$DIRTY"
RES="$REPO/results/bench/$LABEL"
if [[ -e "$RES" ]]; then echo "bench.sh: $RES exists; pick another --label" >&2; exit 2; fi
mkdir -p "$RES/build" "$RES/runs" "$CACHE"
START_ALL=$(date +%s.%N)
echo "== results: $RES"
"${TOOL[@]}" event --dir "$RES" type=env "ld_preload=$( [[ -f "$SHIM" ]] && echo "$SHIM" )" "gmp=$GMP" "cpus=$CPUS" "pin=$PIN" \
    "reps_scale=$REPS_SCALE" "cases=$CASES" "skip=$SKIP" "load=$(cut -d' ' -f1-3 /proc/loadavg)" "gc_flags=${GC_FLAGS[*]}" "gc_heap=$GC_HEAP"

LOAD1="$(cut -d' ' -f1 /proc/loadavg)"
if awk -v l="$LOAD1" 'BEGIN { exit !(l > 4.0) }'; then
    echo "bench.sh: WARNING: load average $LOAD1 before the run; timings will be noisy" >&2
fi

declare -A BIN_PATH
build_binary() {
    local b="$1" src="$REPO/tests/bench/bench_$b.codon" key cached
    key="$("${TOOL[@]}" cachekey --binary "$b")"
    cached="$CACHE/bench_$b-$key"
    if [[ $REBUILD -eq 0 && -x "$cached" && -f "$cached.build_s" ]]; then
        echo "== build $b: cached ($key, built in $(cat "$cached.build_s")s)"
        "${TOOL[@]}" event --dir "$RES" type=build "binary=$b" cached=true "key=$key" "wall_s=$(cat "$cached.build_s")" \
            "log=build/$b.log"
        cp "$cached.log" "$RES/build/$b.log" 2>/dev/null || true
    else
        local tmp t0 t1 rc load_before
        tmp="$(mktemp -d "${TMPDIR:-/tmp}/sequre-bench-build.XXXXXX")"
        load_before="$(cut -d' ' -f1-3 /proc/loadavg)"
        echo "== build $b (release) ..."
        t0=$(date +%s.%N)
        # Flags must come before the input file (after it they are program arguments: a debug build).
        (cd "$tmp" && sequre build -release "$src" -o "$tmp/bench_$b") > "$RES/build/$b.log" 2>&1
        rc=$?
        t1=$(date +%s.%N)
        local wall; wall="$(awk -v a="$t0" -v b="$t1" 'BEGIN { printf "%.2f", b - a }')"
        if [[ $rc -ne 0 || ! -x "$tmp/bench_$b" ]]; then
            echo "bench.sh: build of bench_$b failed (exit $rc); see $RES/build/$b.log" >&2
            tail -20 "$RES/build/$b.log" >&2
            "${TOOL[@]}" event --dir "$RES" type=build "binary=$b" cached=false "key=$key" "wall_s=$wall" "rc=$rc" ok=false
            rm -rf "$tmp"
            return 1
        fi
        mv "$tmp/bench_$b" "$cached"
        echo "$wall" > "$cached.build_s"
        cp "$RES/build/$b.log" "$cached.log"
        rm -rf "$tmp"
        echo "== build $b: ${wall}s"
        "${TOOL[@]}" event --dir "$RES" type=build "binary=$b" cached=false "key=$key" "wall_s=$wall" "rc=$rc" ok=true \
            "log=build/$b.log" "load_before=$load_before" "load_after=$(cut -d' ' -f1-3 /proc/loadavg)" \
            "size_bytes=$(stat -c %s "$cached")"
    fi
    BIN_PATH[$b]="$cached"
}

for b in "${NEEDED[@]}"; do
    build_binary "$b" || { "${TOOL[@]}" manifest --dir "$RES" --label "$LABEL" --argv "$*"; exit 1; }
done

INCOMPLETE=0
run_segment() {
    local b="$1" seg="$2" cases="$3" limit="$4"
    local dir="$RES/runs/$seg" wd t0 t1 rc load_before load_after
    mkdir -p "$dir"
    wd="$(mktemp -d "${TMPDIR:-/tmp}/sequre-bench-run.XXXXXX")"
    local args=(--out="$dir" --reps-scale="$REPS_SCALE" --cases="$cases" --pin="$PIN" "${GC_FLAGS[@]}")
    [[ "$b" != "btp" ]] && args=(--local --skip-mhe-setup "${args[@]}")
    local pin=()
    [[ -n "$CPUS" ]] && pin=(taskset -c "$CPUS")
    local preload=()
    [[ -f "$SHIM" ]] && preload=("LD_PRELOAD=$SHIM")
    [[ -n "$GC_HEAP" && "$GC_HEAP" != "none" ]] && preload+=("GC_INITIAL_HEAP_SIZE=$GC_HEAP")
    load_before="$(cut -d' ' -f1-3 /proc/loadavg)"
    echo "== run $seg (timeout ${limit}s, load $load_before)"
    t0=$(date +%s.%N)
    (cd "$wd" && env "${preload[@]}" "SEQURE_GMP_PATH=$GMP" \
        timeout --kill-after=30 "$limit" "${pin[@]}" "${BIN_PATH[$b]}" "${args[@]}") > "$dir/stdout.log" 2>&1
    rc=$?
    t1=$(date +%s.%N)
    load_after="$(cut -d' ' -f1-3 /proc/loadavg)"
    rm -rf "$wd"
    local wall; wall="$(awk -v a="$t0" -v b="$t1" 'BEGIN { printf "%.2f", b - a }')"
    local complete=true
    if ! "${TOOL[@]}" check --dir "$dir" --binary "$b" --cases "$cases" > "$dir/check.json"; then
        complete=false
        INCOMPLETE=1
        echo "bench.sh: $seg INCOMPLETE (exit $rc): $(cat "$dir/check.json")" >&2
        tail -5 "$dir/stdout.log" >&2
    fi
    echo "== run $seg: ${wall}s, exit $rc, complete=$complete, load after $load_after"
    "${TOOL[@]}" event --dir "$RES" type=run "segment=$seg" "binary=$b" "rc=$rc" "wall_s=$wall" "timeout_s=$limit" \
        "timed_out=$([[ $rc -eq 124 || $rc -eq 137 ]] && echo true || echo false)" "complete=$complete" \
        "load_before=$load_before" "load_after=$load_after" "cpus=$CPUS" "cases=$cases"
}

for b in "${NEEDED[@]}"; do
    main_cases="$(echo "$PLAN" | awk -v b="$b" '$1 == b && $2 == "main" { print $3 }')"
    [[ -n "$main_cases" ]] && run_segment "$b" "$b.main" "$main_cases" "$TIMEOUT"
done
for b in "${NEEDED[@]}"; do
    while read -r _ _ iso; do
        [[ -z "$iso" ]] && continue
        run_segment "$b" "$b.${iso%\$}" "$iso" "$ISO_TIMEOUT"
    done < <(echo "$PLAN" | awk -v b="$b" '$1 == b && $2 == "iso"')
done

"${TOOL[@]}" manifest --dir "$RES" --label "$LABEL" --argv "run --cases '$CASES' --skip '$SKIP' --reps-scale $REPS_SCALE --binaries $BINS --cpus '$CPUS' --pin '$PIN' --gc-heap '$GC_HEAP' ${GC_FLAGS[*]}"
echo "== summary"
"${TOOL[@]}" summary --dir "$RES"
END_ALL=$(date +%s.%N)
echo "== total wall $(awk -v a="$START_ALL" -v b="$END_ALL" 'BEGIN { printf "%.1f", b - a }')s; results in $RES"
exit $INCOMPLETE
