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
#
# Output structure:
#   certs/
#     ca.pem          - CA certificate (shared by all parties)
#     ca-key.pem      - CA private key (keep secure, not distributed)
#     cp0.pem         - Party 0 (trusted dealer) certificate
#     cp0-key.pem     - Party 0 private key
#     cp1.pem         - Party 1 certificate
#     cp1-key.pem     - Party 1 private key
#     ...

set -euo pipefail

NUM_PARTIES="${1:-3}"
CERT_DIR="${2:-./certs}"
CA_DAYS="${SEQURE_CA_DAYS:-1095}"
CERT_DAYS="${SEQURE_CERT_DAYS:-365}"
RSA_BITS="${SEQURE_RSA_BITS:-2048}"

mkdir -p "$CERT_DIR"

echo "=== Generating Sequre MPC certificates for ${NUM_PARTIES} parties ==="
echo "[warning] Development/test helper only. Do not use as production PKI workflow."

# Generate CA key and self-signed certificate
echo "[1/${NUM_PARTIES}+1] Generating CA..."
openssl req -x509 \
    -newkey "rsa:${RSA_BITS}" \
    -keyout "${CERT_DIR}/ca-key.pem" \
    -out "${CERT_DIR}/ca.pem" \
    -days "$CA_DAYS" \
    -nodes \
    -subj "/CN=SequreMPC-CA" \
    2>/dev/null

# Generate per-party certificates signed by the CA
for pid in $(seq 0 $((NUM_PARTIES - 1))); do
    step=$((pid + 2))
    total=$((NUM_PARTIES + 1))
    echo "[${step}/${total}] Generating certificate for CP${pid}..."

    # Generate key + CSR
    openssl req \
        -newkey "rsa:${RSA_BITS}" \
        -keyout "${CERT_DIR}/cp${pid}-key.pem" \
        -out "${CERT_DIR}/cp${pid}.csr" \
        -nodes \
        -subj "/CN=cp${pid}" \
        2>/dev/null

    # Sign with CA
    openssl x509 -req \
        -in "${CERT_DIR}/cp${pid}.csr" \
        -CA "${CERT_DIR}/ca.pem" \
        -CAkey "${CERT_DIR}/ca-key.pem" \
        -CAcreateserial \
        -out "${CERT_DIR}/cp${pid}.pem" \
        -days "$CERT_DAYS" \
        2>/dev/null

    # Clean up CSR
    rm -f "${CERT_DIR}/cp${pid}.csr"
done

# Clean up serial file
rm -f "${CERT_DIR}/ca.srl"

# Restrict permissions on private keys
chmod 600 "${CERT_DIR}"/*-key.pem

echo "=== Done. Certificates written to ${CERT_DIR}/ ==="
echo ""
echo "Files:"
ls -la "${CERT_DIR}/"
echo ""
echo "Distribute to each party:"
echo "  - ca.pem (all parties)"
echo "  - cp<N>.pem + cp<N>-key.pem (to party N only)"
echo "  - Do not distribute ca-key.pem (only needed for issuing new certs)"
