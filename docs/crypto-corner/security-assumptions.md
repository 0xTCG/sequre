# Security Assumptions

This page documents the threat model and cryptographic hardness assumptions that underpin Sequre and Shechi.

## Threat model

### Semi-honest (honest-but-curious)

Sequre assumes all parties follow the protocol correctly but may try to learn additional information from the messages they observe. This is the standard **semi-honest** model.

- **No malicious behavior**: Parties do not send forged messages, deviate from the protocol, or abort strategically.
- **Collusion**: Privacy holds against any single party, and against any coalition of compute parties that leaves at least one compute party out, as long as the trusted dealer (CP0) is not part of it. CP0 deals the correlated randomness (Beaver triples, masks), so CP0 together with any one compute party can recover values that were opened against that randomness. The MHE collective secret key is shared among the compute parties only (CP0 holds a zero share), so a collective decryption needs all of them.

### Trusted dealer (CP0)

Party 0 (CP0) acts as a **trusted dealer** during the offline phase:

- Generates and distributes **Beaver triples** (correlated randomness) for MPC multiplication
- Does **not** participate in online computation (holds zero shares / zero secret-key polynomial)

In production deployments, the trusted dealer can be replaced by an MPC preprocessing protocol or a hardware enclave.

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

For the MPC (non-HE) path, values are additively shared over a prime field (or a power-of-two ring):

$$x = x_1 + x_2 + \cdots + x_N \pmod{p}$$

Shares and masks are drawn uniformly from the whole field with a cryptographically secure PRG, seeded pairwise between the parties over TLS. The shares seen by a coalition that misses at least one party (subject to the dealer caveat above) are therefore computationally indistinguishable from uniform: security rests on the PRG, not on perfect (information-theoretic) secrecy.

## Noise and masks in the MHE protocols

What each protocol reveals to the party that combines the shares (the hub):

- **Collective decryption.** Each compute party sends the hub $s_i \cdot c_1 + e_i$, with Gaussian noise $e_i$ of standard deviation `SEQURE_SIGMA_SMUDGE` (default $2^{10}$; values below 3.2 are rejected). Without noise, the hub could compute every key share as $s_i = \text{share} / c_1$; with it, recovering $s_i$ is a ring-LWE problem. The noise does not *flood* the ciphertext noise statistically, so a decrypted result still carries some information about the ciphertext's own noise (the IND-CPA-D setting). Flooding needs a standard deviation of about $2^{20}$ or more together with a larger scale, because the decryption error grows in proportion to it: the measured maximum error at the default scale $2^{34}$ is about $5 \cdot 10^{-5}$ at $2^{10}$, $5 \cdot 10^{-4}$ at $2^{14}$, and $3 \cdot 10^{-2}$ at $2^{20}$.
- **HE → MPC switching.** Before the hub decrypts, each party adds a uniformly random mask, bounded by $2^{32}$ in each real and imaginary part of every slot; only the hub receives the masked result. A slot value of magnitude $B$ is hidden from the hub only up to a statistical distance of about $B / 2^{33}$ (about $2^{-23}$ for values up to 1000), not $2^{-\lambda}$: the masks go through the floating-point encoder, which limits their size.
- **MPC → HE switching.** The parties open $x - R$, where $R$ is a sum of masks of $64 + f - \log_2 \Delta - b$ bits ($f$ the fixed-point fractional bits, $b$ the bit length of the number of compute parties): 60 bits for the default three parties (two compute parties) at scale $2^{34}$. A value of magnitude $B$ is therefore hidden only up to a statistical distance of about $B \cdot 2^{f} / 2^{60} = B \cdot 2^{-28}$, again a limit of the floating-point encoder.
- **Collective bootstrapping (refresh).** Partial decryptions are masked with random integers of $\lambda + \log_2 \Delta$ bits in the coefficient domain ($\lambda = 128$), which hide the plaintext and its noise statistically.

Statistically hiding masks for both switching directions are larger than the floating-point encoder can carry. Lattiseq's exact encoder ([CKKS Operations](../deep-dive-lattiseq/ckks-operations.md#exact-encoding)) carries them; moving the switching protocols onto it is planned.

## CKKS approximate precision

CKKS is an **approximate** HE scheme — decrypted values have bounded error:

$$|\hat{m}_i - m_i| \leq \frac{\|e\|_\infty}{\Delta}$$

where $\Delta$ is the encoding scale. Sequre/Shechi manages scale and level tracking automatically, but users should be aware that:

- Each multiplication roughly doubles the error
- Rescaling removes one level but restores the scale
- Bootstrapping (refresh) restores levels at the cost of additional noise

## Parameter selection guidelines

| Workload | Recommended | Levels | Why |
|---|---|---|---|
| Light (few multiplications) | `PN12QP109` | 2 | Fastest, smallest keys |
| Medium | `PN14QP438` | 10 | Good balance (Sequre/Shechi default) |
| Heavy (deep circuits) | `PN15QP880` | 18 | More levels before bootstrap |
| Very heavy | `PN16QP1761` | 34 | Maximum depth |

## Next steps

- **[Schemes and Protocols](schemes-protocols.md)** — Mathematical details of each protocol.
- **[Configuration](../api/configuration.md)** — Security-related settings.
- **[CKKS Operations](../deep-dive-lattiseq/ckks-operations.md)** — How the scheme is implemented.
