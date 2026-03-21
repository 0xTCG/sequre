# Security Assumptions

This page documents the threat model and cryptographic hardness assumptions that underpin Sequre and Shechi.

## Threat model

### Semi-honest (honest-but-curious)

Sequre assumes all parties follow the protocol correctly but may try to learn additional information from the messages they observe. This is the standard **semi-honest** model.

- **No malicious behavior**: Parties do not send forged messages, deviate from the protocol, or abort strategically.
- **No collusion beyond threshold**: For $N$ compute parties, security holds as long as no single party can reconstruct the full secret. In the additive secret-sharing setting, all $N$ parties must collude to break privacy.

### Trusted dealer (CP0)

Party 0 (CP0) acts as a **trusted dealer** during the offline phase:

- Generates and distributes **Beaver triples** (correlated randomness) for MPC multiplication
- Generates shared PRNG seeds for common reference polynomials
- Does **not** participate in online computation (holds zero shares / zero secret-key polynomial)

In production deployments, the trusted dealer can be replaced by an MPC preprocessing protocol or a hardware enclave.

### Network assumptions

- **Authenticated channels**: All communication is assumed to be authenticated (no man-in-the-middle).
- **Synchronous rounds**: The protocols assume synchronous communication — all parties complete each round before the next begins.
- **Star topology**: Parties communicate through a hub (typically the trusted dealer or a designated CP).

## Hardness assumptions

### Ring Learning with Errors (RLWE)

The CKKS scheme (and all HE in Shechi) is based on the hardness of the **RLWE problem** over the cyclotomic ring $R = \mathbb{Z}[X]/(X^N+1)$:

!!! abstract "RLWE assumption"
    Given polynomials $(a_i, b_i = a_i \cdot s + e_i) \in R_q^2$ where $s \leftarrow \chi_s$ and $e_i \leftarrow \chi_e$ are sampled from secret and error distributions, it is computationally infeasible to distinguish $(a_i, b_i)$ from uniform random pairs in $R_q^2$.

Parameters:

- $N$ (ring degree) — controls security level and slot count. Sequre uses $N \in \{4096, 8192, 16384, 32768, 65536\}$
- $q$ (ciphertext modulus) — product of RNS primes. Larger $q$ allows more levels but weakens security for fixed $N$
- $\sigma$ (Gaussian width) — standard deviation of the error distribution, typically 3.2

Security is estimated using the [Lattice Estimator](https://github.com/malb/lattice-estimator). Sequre's default parameters target **128-bit security**.

### Decisional composite residuosity (not used)

Sequre does **not** use Paillier or other DCR-based schemes. All HE is RLWE-based.

## Additive secret sharing security

For the MPC (non-HE) path, security relies on **information-theoretic** properties of additive sharing:

$$x = x_1 + x_2 + \cdots + x_N \pmod{p}$$

Each share $x_i$ is uniformly random given any proper subset of shares. This provides **perfect secrecy** against up to $N-1$ colluding parties (in the honest-majority or full-threshold variant used by Sequre).

## Smudging noise

During protocol switching (HE → MPC), each party reveals $\text{sk}_i \cdot c_1 + \text{mask}_i + e_i$. The smudging noise $e_i$ is sampled from a Gaussian with width controlled by `LATTISEQ_SIGMA_SMUDGE`. This prevents leakage from the partial decryption beyond what the output itself reveals:

$$\text{Statistical distance} \leq 2^{-\lambda}$$

where $\lambda$ is the security parameter (128 bits).

## CKKS approximate precision

CKKS is an **approximate** HE scheme — decrypted values have bounded error:

$$|\hat{m}_i - m_i| \leq \frac{\|e\|_\infty}{\Delta}$$

where $\Delta$ is the encoding scale. Sequre manages scale and level tracking automatically, but users should be aware that:

- Each multiplication roughly doubles the error
- Rescaling removes one level but restores the scale
- Bootstrapping (refresh) restores levels at the cost of additional noise

## Parameter selection guidelines

| Workload | Recommended | Levels | Why |
|---|---|---|---|
| Light (few multiplications) | `PN12QP109` | 2 | Fastest, smallest keys |
| Medium | `PN14QP438` | 10 | Good balance (Sequre default) |
| Heavy (deep circuits) | `PN15QP880` | 18 | More levels before bootstrap |
| Very heavy | `PN16QP1761` | 34 | Maximum depth |

## Next steps

- **[Schemes and Protocols](schemes-protocols.md)** — Mathematical details of each protocol.
- **[Configuration](../api/configuration.md)** — Security-related settings.
- **[CKKS Operations](../deep-dive-lattiseq/ckks-operations.md)** — How the scheme is implemented.
