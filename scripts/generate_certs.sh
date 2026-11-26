#!/usr/bin/env bash
#
# Generates mTLS certificates for Sequre MPC parties (DEVELOPMENT / TESTING ONLY).
#
# WARNING:
#   This script is not intended for production certificate lifecycle management.
#   In production, provision ca.pem / cpN.pem / cpN-key.pem via private PKI pipeline.
#
# Usage: ./scripts/generate_certs.sh [num_parties] [output_dir]
#   num_parties: Total number of computing parties including trusted dealer (default: 3)
#   output_dir:  Directory to write certificates to (default: ./certs)
#
# Optional env vars:
#   SEQURE_CA_DAYS=<days>    CA certificate validity (default: 1095)
#   SEQURE_CERT_DAYS=<days>  Party certificate validity (default: 365)
#   SEQURE_RSA_BITS=<bits>   RSA key size (default: 2048)
#   SEQURE_PARTY_CERT_NAME=<pattern>
#                            The name each party certificate carries, "{pid}" replaced by the party ID (default:
#                            cp{pid}). The parties check it at every TLS handshake, so run them with the same value.
#
# Each party certificate names its party twice: as a subjectAltName DNS entry and as the subject CN. A party accepts a
# peer only if the peer's certificate comes from the CA and names the party the peer connects as.
#
# The CA is made for this run only: its private key signs the party certificates and is then deleted, so no CA key is
# left anywhere. To add or replace a party, rerun the script for all parties. Private keys are created readable by
# their owner only.
#
# Output structure:
#   certs/
#     ca.pem          - CA certificate (shared by all parties)
#     cp0.pem         - Party 0 (trusted dealer) certificate
#     cp0-key.pem     - Party 0 private key
#     cp1.pem         - Party 1 certificate
#     cp1-key.pem     - Party 1 private key
#     ...

set -euo pipefail

DEFAULT_NAME_PATTERN='cp{pid}'
NUM_PARTIES="${1:-3}"
CERT_DIR="${2:-./certs}"
CA_DAYS="${SEQURE_CA_DAYS:-1095}"
CERT_DAYS="${SEQURE_CERT_DAYS:-365}"
RSA_BITS="${SEQURE_RSA_BITS:-2048}"
NAME_PATTERN="${SEQURE_PARTY_CERT_NAME:-$DEFAULT_NAME_PATTERN}"

die() { echo "generate_certs.sh: $*" >&2; exit 1; }

[[ "$NUM_PARTIES" =~ ^[1-9][0-9]*$ ]] || die "num_parties must be a positive integer (got '$NUM_PARTIES')"
[[ "$NAME_PATTERN" == *"{pid}"* ]] || die "SEQURE_PARTY_CERT_NAME must contain {pid}, so that each party gets a name of its own (got '$NAME_PATTERN')"
party_name() { echo "${NAME_PATTERN//\{pid\}/$1}"; }
for pid in $(seq 0 $((NUM_PARTIES - 1))); do
    [[ "$(party_name "$pid")" =~ ^[A-Za-z0-9]([A-Za-z0-9.-]*[A-Za-z0-9])?$ ]] \
        || die "the name for CP${pid} ('$(party_name "$pid")', from SEQURE_PARTY_CERT_NAME) is not a DNS name"
done
command -v openssl > /dev/null || die "openssl not found"

mkdir -p "$CERT_DIR"
umask 077  # every file below starts readable by its owner only; the certificates are opened up at the end
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

# Runs openssl with its progress output captured; if it fails, shows that output and stops.
run_openssl() {
    if ! openssl "$@" > "$WORK/openssl.log" 2>&1; then
        echo "generate_certs.sh: openssl $1 failed:" >&2
        cat "$WORK/openssl.log" >&2
        exit 1
    fi
}

echo "=== Generating Sequre MPC certificates for ${NUM_PARTIES} parties ==="
echo "[warning] Development/test helper only. Do not use as production PKI workflow."

# A CA key left by an earlier version of this script belongs to the CA that is replaced below.
if [[ -e "${CERT_DIR}/ca-key.pem" ]]; then
    rm -f "${CERT_DIR}/ca-key.pem"
    echo "Removed ${CERT_DIR}/ca-key.pem (an earlier run's CA key; the CA key is no longer kept)."
fi

# Generate the CA key (in a private temporary directory) and self-signed certificate
echo "[1/$((NUM_PARTIES + 1))] Generating CA..."
run_openssl req -x509 \
    -newkey "rsa:${RSA_BITS}" \
    -keyout "$WORK/ca-key.pem" \
    -out "${CERT_DIR}/ca.pem" \
    -days "$CA_DAYS" \
    -nodes \
    -subj "/CN=SequreMPC-CA"

# Generate per-party certificates signed by the CA
for pid in $(seq 0 $((NUM_PARTIES - 1))); do
    name="$(party_name "$pid")"
    echo "[$((pid + 2))/$((NUM_PARTIES + 1))] Generating certificate for CP${pid} (${name})..."

    # Generate key + CSR (a new key file, so that it gets the owner-only mode even if an old one was readable)
    rm -f "${CERT_DIR}/cp${pid}-key.pem"
    run_openssl req \
        -newkey "rsa:${RSA_BITS}" \
        -keyout "${CERT_DIR}/cp${pid}-key.pem" \
        -out "$WORK/cp${pid}.csr" \
        -nodes \
        -subj "/CN=${name}"

    # Sign with the CA, adding the party's name as a subjectAltName. Each party is a TLS server for some channels and
    # a client for others.
    cat > "$WORK/cp${pid}.ext" <<EOF
basicConstraints = critical, CA:FALSE
keyUsage = critical, digitalSignature, keyEncipherment
extendedKeyUsage = serverAuth, clientAuth
subjectKeyIdentifier = hash
authorityKeyIdentifier = keyid, issuer
subjectAltName = DNS:${name}
EOF
    run_openssl x509 -req \
        -in "$WORK/cp${pid}.csr" \
        -CA "${CERT_DIR}/ca.pem" \
        -CAkey "$WORK/ca-key.pem" \
        -CAserial "$WORK/ca.srl" \
        -CAcreateserial \
        -extfile "$WORK/cp${pid}.ext" \
        -out "${CERT_DIR}/cp${pid}.pem" \
        -days "$CERT_DAYS"
    chmod 644 "${CERT_DIR}/cp${pid}.pem"
done
chmod 644 "${CERT_DIR}/ca.pem"
# The CA key, the CSRs and the serial file go with $WORK on exit.

echo "=== Done. Certificates written to ${CERT_DIR}/ ==="
echo ""
echo "Files:"
ls -la "${CERT_DIR}/"
echo ""
echo "Distribute to each party:"
echo "  - ca.pem (all parties)"
echo "  - cp<N>.pem + cp<N>-key.pem (to party N only)"
echo "The CA's private key was deleted after signing: to add or replace a party, rerun this script for all parties."
if [[ "$NAME_PATTERN" != "$DEFAULT_NAME_PATTERN" ]]; then
    echo "The certificates name the parties ${NAME_PATTERN}: run every party with SEQURE_PARTY_CERT_NAME='${NAME_PATTERN}'."
fi
