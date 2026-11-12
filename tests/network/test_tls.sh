#!/usr/bin/env bash
#
# Distributed-mode tests: every party runs as a process of its own on 127.0.0.1 and the parties connect over TCP, with
# TLS, as in a deployment (the test runner's groups fork the parties and connect them over UNIX sockets, without TLS).
#
# Usage: tests/network/test_tls.sh [--binary FILE] [--work DIR] [--list] [SCENARIO ...]
#   Builds tests/network/tls_parties.codon with `sequre build` (or runs the --binary given), makes certificates in the
#   work directory (default: a new temporary directory), runs the scenarios (default: all) and prints one line per
#   scenario. Exit status: the number of failed scenarios (2 for a usage or setup error). Each party's output is in
#   <work>/<scenario>/cp<pid>.log.
#
# Scenarios (MHE setup on in the a- and e- runs, off in the others):
#   a-san-3, a-san-4  certificates from scripts/generate_certs.sh (each names its party as a subjectAltName and as the
#                     CN), 3 and 4 parties: the run completes and the revealed product is right
#   a-cn-3, a-cn-4    the same with CN-only certificates (no subjectAltName), as generate_certs.sh made them before
#   b-server          CP1 presents CP2's certificate and key: CP0 (the server of that channel) rejects it, and CP1
#                     learns it was rejected
#   b-client          CP0 presents CP1's certificate and key: CP1 (the client) rejects it
#   c-other-ca        CP1 presents a certificate named cp1 from another CA: CP0 rejects it
#   d-names           other names (SEQURE_PARTY_CERT_NAME=party-{pid}.sequre.test, certificates that carry them): the
#                     run completes
#   d-wrong-names     those certificates while the parties expect the default names: rejected
#   d-no-pid          SEQURE_PARTY_CERT_NAME without {pid}: every party refuses to start
#   e-no-tls          SEQURE_USE_TLS=0, plain TCP: the run completes
#   f-dead-peer       CP2 exits without closing its connections while CP0 sends it 100 messages of 800 KB: CP0's TLS
#                     send fails and stops CP0 (a failed SSL_write used to count as a success)
#
# Needs bash, openssl, and the parties' ports free on 127.0.0.1: 9001-9006 (COMMUNICATION_PORT + 1 to + 6). The program
# runs MHE code, which needs Sequre's alignment shim on some x86-64 machines: if LD_PRELOAD is unset, the script
# preloads $HOME/.sequre/lib/sequre_align64.so when it exists, as the sequre launcher does. About 3 minutes.

set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ALL_SCENARIOS=(a-san-3 a-san-4 a-cn-3 a-cn-4 b-server b-client c-other-ca d-names d-wrong-names d-no-pid e-no-tls
               f-dead-peer)
PORTS=(9001 9002 9003 9004 9005 9006)
PASS_LIMIT=180    # seconds a run that should complete may take
REJECT_LIMIT=60   # seconds a run that should stop may take
GRACE=5           # once a party has failed, seconds before the parties still running are stopped

usage() { awk 'NR > 1 && /^#/ { sub(/^# ?/, ""); print; next } NR > 1 { exit }' "${BASH_SOURCE[0]}"; }

BIN="" WORK="" SCENARIOS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --binary) BIN="$2"; shift 2 ;;
        --work) WORK="$2"; shift 2 ;;
        --list) printf '%s\n' "${ALL_SCENARIOS[@]}"; exit 0 ;;
        -h|--help) usage; exit 0 ;;
        -*) echo "test_tls.sh: unknown option $1" >&2; usage >&2; exit 2 ;;
        *) SCENARIOS+=("$1"); shift ;;
    esac
done
[[ ${#SCENARIOS[@]} -gt 0 ]] || SCENARIOS=("${ALL_SCENARIOS[@]}")
for s in "${SCENARIOS[@]}"; do
    [[ " ${ALL_SCENARIOS[*]} " == *" $s "* ]] || { echo "test_tls.sh: unknown scenario $s" >&2; exit 2; }
done

command -v openssl > /dev/null || { echo "test_tls.sh: openssl not found" >&2; exit 2; }

# A party already listening on one of the ports would take the connections (checked without connecting to it)
port_in_use() {
    if command -v ss > /dev/null; then
        [[ -n "$(ss -Hltn "sport = :$1" 2> /dev/null)" ]]
    elif command -v lsof > /dev/null; then
        lsof -nP -iTCP:"$1" -sTCP:LISTEN > /dev/null 2>&1
    else
        return 1
    fi
}
for port in "${PORTS[@]}"; do
    if port_in_use "$port"; then
        echo "test_tls.sh: port $port is in use; the parties need ports ${PORTS[0]}-${PORTS[${#PORTS[@]} - 1]} on 127.0.0.1" >&2
        exit 2
    fi
done

if [[ -z "$WORK" ]]; then
    WORK="$(mktemp -d "${TMPDIR:-/tmp}/sequre-tls-test.XXXXXX")"
fi
mkdir -p "$WORK"
WORK="$(cd "$WORK" && pwd)"
echo "Work directory: $WORK"

if [[ -z "$BIN" ]]; then
    command -v sequre > /dev/null || { echo "test_tls.sh: sequre is not on PATH (or pass --binary)" >&2; exit 2; }
    echo "Building tests/network/tls_parties.codon ..."
    # Built from the work directory: the launcher deletes the sock.* files of the directory it runs in
    if ! (cd "$WORK" && sequre build "$REPO/tests/network/tls_parties.codon" -o "$WORK/tls_parties" \
            > "$WORK/build.log" 2>&1); then
        echo "test_tls.sh: the build failed, see $WORK/build.log" >&2
        exit 2
    fi
    BIN="$WORK/tls_parties"
fi
[[ -x "$BIN" ]] || { echo "test_tls.sh: $BIN is not an executable" >&2; exit 2; }
BIN="$(cd "$(dirname "$BIN")" && pwd)/$(basename "$BIN")"
if [[ -z "${LD_PRELOAD:-}" && -f "$HOME/.sequre/lib/sequre_align64.so" ]]; then
    export LD_PRELOAD="$HOME/.sequre/lib/sequre_align64.so"
fi

# Certificates for 4 parties: std (generate_certs.sh), other (a second CA, same names), custom (other names), cn (the
# name in the subject CN only, no extensions)
CERTS="$WORK/certs"
make_cn_only_certs() {
    local dir=$1 pid
    mkdir -p "$dir"
    openssl req -x509 -newkey rsa:2048 -keyout "$dir/ca-key.pem" -out "$dir/ca.pem" -days 30 -nodes \
        -subj "/CN=SequreTest-CN-CA" || return 1
    for pid in 0 1 2 3; do
        openssl req -newkey rsa:2048 -keyout "$dir/cp$pid-key.pem" -out "$dir/cp$pid.csr" -nodes -subj "/CN=cp$pid" \
            || return 1
        openssl x509 -req -in "$dir/cp$pid.csr" -CA "$dir/ca.pem" -CAkey "$dir/ca-key.pem" -CAcreateserial \
            -out "$dir/cp$pid.pem" -days 30 || return 1
    done
}
echo "Generating certificates ..."
rm -rf "$CERTS"
mkdir -p "$CERTS"
if ! { bash "$REPO/scripts/generate_certs.sh" 4 "$CERTS/std" > "$CERTS/std.log" 2>&1 \
        && bash "$REPO/scripts/generate_certs.sh" 4 "$CERTS/other" > "$CERTS/other.log" 2>&1 \
        && SEQURE_PARTY_CERT_NAME='party-{pid}.sequre.test' \
            bash "$REPO/scripts/generate_certs.sh" 4 "$CERTS/custom" > "$CERTS/custom.log" 2>&1 \
        && make_cn_only_certs "$CERTS/cn" > "$CERTS/cn.log" 2>&1; }; then
    echo "test_tls.sh: certificate generation failed, see $CERTS/*.log" >&2
    exit 2
fi

# Per-party environment of each scenario (called in the party's subshell with its pid)
env_std() { :; }
env_cn() { export SEQURE_CERT_DIR="$CERTS/cn"; }
env_b_server() { [[ $1 != 1 ]] || export SEQURE_PARTY_CERT_FILE=cp2.pem SEQURE_PARTY_KEY_FILE=cp2-key.pem; }
env_b_client() { [[ $1 != 0 ]] || export SEQURE_PARTY_CERT_FILE=cp1.pem SEQURE_PARTY_KEY_FILE=cp1-key.pem; }
env_c_other_ca() { [[ $1 != 1 ]] || export SEQURE_CERT_DIR="$CERTS/other" SEQURE_CA_CERT_FILE=../std/ca.pem; }
env_d_names() { export SEQURE_CERT_DIR="$CERTS/custom" SEQURE_PARTY_CERT_NAME='party-{pid}.sequre.test'; }
env_d_wrong_names() { export SEQURE_CERT_DIR="$CERTS/custom"; }
env_d_no_pid() { export SEQURE_PARTY_CERT_NAME=party; }
env_e_no_tls() { export SEQURE_USE_TLS=0; }
env_f_dead_peer() { export TLS_CHECK_DEAD_PEER=1; }

# run_parties SCENARIO N LIMIT ENV_FN [ARGS...]: runs parties 0..N-1 in $WORK/SCENARIO, each with the certificates in
# $CERTS/std unless ENV_FN changes its environment, until all have exited, LIMIT seconds have passed, or GRACE seconds
# after the first failure. Parties still running then are stopped (cp<pid>.stopped). Exit codes go to cp<pid>.rc.
run_parties() {
    local name=$1 n=$2 limit=$3 env_fn=$4
    shift 4
    local dir="$WORK/$name" ips p rc finished start failed_at
    local subshells=()
    rm -rf "$dir"
    mkdir -p "$dir"
    ips="127.0.0.1"
    for ((p = 1; p < n; p++)); do ips+=",127.0.0.1"; done
    for ((p = 0; p < n; p++)); do
        (
            cd "$dir" || exit 99
            export SEQURE_CP_IPS="$ips" SEQURE_CERT_DIR="$CERTS/std"
            "$env_fn" "$p"
            "$BIN" "$p" "$@" > "cp$p.log" 2>&1
            echo $? > "cp$p.rc"
        ) 2> /dev/null &  # (the subshell's own notices, such as "Aborted", would clutter the summary)
        subshells+=($!)
    done
    start=$SECONDS
    failed_at=-1
    while :; do
        sleep 0.5
        finished=0
        for ((p = 0; p < n; p++)); do
            [[ -f "$dir/cp$p.rc" ]] || continue
            finished=$((finished + 1))
            rc=$(cat "$dir/cp$p.rc")
            if [[ "$rc" != 0 && $failed_at -lt 0 ]]; then failed_at=$SECONDS; fi
        done
        ((finished == n)) && break
        if ((SECONDS - start >= limit)) || ((failed_at >= 0 && SECONDS - failed_at >= GRACE)); then
            for ((p = 0; p < n; p++)); do
                if [[ ! -f "$dir/cp$p.rc" ]]; then
                    touch "$dir/cp$p.stopped"
                    pkill -TERM -P "${subshells[$p]}" 2> /dev/null
                fi
            done
            sleep 1
            for ((p = 0; p < n; p++)); do pkill -KILL -P "${subshells[$p]}" 2> /dev/null; done
            break
        fi
    done
    wait
}

PROBLEMS=()
party_rc() { cat "$WORK/$1/cp$2.rc" 2> /dev/null || echo "?"; }
# Case-insensitive: OpenSSL's wording of a reason differs in case between versions ("Hostname mismatch" in 1.1)
log_has() { grep -qiF -- "$3" "$WORK/$1/cp$2.log"; }
# The first line of a party's output containing a text, without terminal colour codes
log_line() { grep -F -m1 -- "$3" "$WORK/$1/cp$2.log" | sed 's/\x1b\[[0-9;]*m//g'; }

# expect_complete SCENARIO N: every party exited with status 0 and got to the end; every compute party checked the
# product and printed no FAIL
expect_complete() {
    local name=$1 n=$2 p
    for ((p = 0; p < n; p++)); do
        [[ "$(party_rc "$name" $p)" == 0 ]] || PROBLEMS+=("CP$p exited with $(party_rc "$name" $p)$([[ -f "$WORK/$name/cp$p.stopped" ]] && echo ' (stopped)')")
        log_has "$name" $p "CP$p: TLS-CHECK DONE" || PROBLEMS+=("CP$p did not get to the end")
        if ((p > 0)); then
            log_has "$name" $p "CP$p: ok share from CP1" || PROBLEMS+=("CP$p printed no ok line")
            ! log_has "$name" $p "CP$p: FAIL" || PROBLEMS+=("CP$p printed a FAIL line")
        fi
    done
}

# expect_stopped SCENARIO N: no party got to the end
expect_stopped() {
    local name=$1 n=$2 p
    for ((p = 0; p < n; p++)); do
        ! log_has "$name" $p "TLS-CHECK DONE" || PROBLEMS+=("CP$p got to the end of the run")
    done
}

# expect_any SCENARIO PARTY TEXT...: at least one TEXT is in PARTY's output
expect_any() {
    local name=$1 p=$2 text
    shift 2
    for text in "$@"; do
        log_has "$name" $p "$text" && return
    done
    PROBLEMS+=("CP$p's output has none of: $(printf '"%s" ' "$@")")
}

# expect_failure SCENARIO PARTY TEXT...: PARTY failed by itself (not stopped by this script), with every TEXT in its
# output
expect_failure() {
    local name=$1 p=$2 text
    shift 2
    if [[ -f "$WORK/$name/cp$p.stopped" || "$(party_rc "$name" $p)" == 0 ]]; then
        PROBLEMS+=("CP$p did not fail by itself (exit $(party_rc "$name" $p)$([[ -f "$WORK/$name/cp$p.stopped" ]] && echo ', stopped'))")
    fi
    for text in "$@"; do
        log_has "$name" $p "$text" || PROBLEMS+=("CP$p's output lacks \"$text\"")
    done
}

run_scenario() {
    local s=$1
    case "$s" in
        a-san-3) run_parties "$s" 3 $PASS_LIMIT env_std; expect_complete "$s" 3 ;;
        a-san-4) run_parties "$s" 4 $PASS_LIMIT env_std; expect_complete "$s" 4 ;;
        a-cn-3) run_parties "$s" 3 $PASS_LIMIT env_cn; expect_complete "$s" 3 ;;
        a-cn-4) run_parties "$s" 4 $PASS_LIMIT env_cn; expect_complete "$s" 4 ;;
        b-server)
            run_parties "$s" 3 $REJECT_LIMIT env_b_server --skip-mhe-setup
            expect_stopped "$s" 3
            expect_failure "$s" 0 "TLS handshake with CP1 at CP0 (server side" \
                "CP1's certificate was rejected: hostname mismatch" 'names "cp1"'
            # TLS 1.3 completes the client's handshake first, so CP1 hears of the rejection at the confirmation;
            # under TLS 1.2 its own handshake fails
            expect_failure "$s" 1 "TLS handshake with CP0 at CP1 (client side"
            expect_any "$s" 1 "CP0 did not confirm the handshake" "CP0 rejected the handshake" ;;
        b-client)
            run_parties "$s" 3 $REJECT_LIMIT env_b_client --skip-mhe-setup
            expect_stopped "$s" 3
            expect_failure "$s" 1 "TLS handshake with CP0 at CP1 (client side" \
                "CP0's certificate was rejected: hostname mismatch" 'names "cp0"'
            expect_failure "$s" 0 "TLS handshake with CP1 at CP0 (server side" "CP1 rejected the handshake" ;;
        c-other-ca)
            run_parties "$s" 3 $REJECT_LIMIT env_c_other_ca --skip-mhe-setup
            expect_stopped "$s" 3
            expect_failure "$s" 0 "TLS handshake with CP1 at CP0 (server side" "CP1's certificate was rejected:" \
                "expected a certificate from the CA in" ;;
        d-names) run_parties "$s" 3 $PASS_LIMIT env_d_names --skip-mhe-setup; expect_complete "$s" 3 ;;
        d-wrong-names)
            run_parties "$s" 3 $REJECT_LIMIT env_d_wrong_names --skip-mhe-setup
            expect_stopped "$s" 3
            expect_failure "$s" 1 "TLS handshake with CP0 at CP1 (client side" \
                "CP0's certificate was rejected: hostname mismatch" 'names "cp0"' ;;
        d-no-pid)
            run_parties "$s" 3 $REJECT_LIMIT env_d_no_pid --skip-mhe-setup
            expect_stopped "$s" 3
            expect_failure "$s" 0 'SEQURE_PARTY_CERT_NAME ("party") must contain {pid}'
            expect_failure "$s" 2 'SEQURE_PARTY_CERT_NAME ("party") must contain {pid}' ;;
        e-no-tls)
            run_parties "$s" 3 $PASS_LIMIT env_e_no_tls
            expect_complete "$s" 3
            ! grep -qF "TLS handshake" "$WORK/$s"/cp*.log || PROBLEMS+=("a TLS handshake ran with SEQURE_USE_TLS=0") ;;
        f-dead-peer)
            run_parties "$s" 3 $REJECT_LIMIT env_f_dead_peer --skip-mhe-setup
            expect_failure "$s" 0 "TLS connection broken: sent"
            ! log_has "$s" 0 "CP0: sent 100 messages" || PROBLEMS+=("CP0's sends to the exited CP2 all went through")
            [[ "$(party_rc "$s" 2)" == 0 ]] || PROBLEMS+=("CP2 did not exit with 0 ($(party_rc "$s" 2))") ;;
    esac
}

# The line of a rejection scenario's output worth showing in the summary
evidence() {
    case "$1" in
        b-server|c-other-ca) log_line "$1" 0 "TLS handshake with CP1 at CP0" ;;
        b-client|d-wrong-names) log_line "$1" 1 "TLS handshake with CP0 at CP1" ;;
        d-no-pid) log_line "$1" 0 "must contain" ;;
        f-dead-peer) log_line "$1" 0 "TLS connection broken" ;;
    esac
}

FAILED=0
for s in "${SCENARIOS[@]}"; do
    PROBLEMS=()
    begin=$SECONDS
    run_scenario "$s"
    if [[ ${#PROBLEMS[@]} -eq 0 ]]; then
        printf 'PASS  %-14s (%ds)\n' "$s" $((SECONDS - begin))
    else
        printf 'FAIL  %-14s (%ds): %s\n' "$s" $((SECONDS - begin)) "$(IFS=';'; echo "${PROBLEMS[*]}")"
        FAILED=$((FAILED + 1))
    fi
    line="$(evidence "$s")"
    [[ -z "$line" ]] || echo "      $line"
done
echo "$((${#SCENARIOS[@]} - FAILED)) of ${#SCENARIOS[@]} scenarios passed; party output in $WORK"
exit $FAILED
