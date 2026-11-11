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

## Decor and bitwise (XOR) sharing

Sequre's nonlinear functions ([Nonlinear Functions](../user-guide/nonlinear-functions.md)) and its ring comparisons,
division and square root add two kinds of messages to the MPC path. Neither needs a new assumption.

**Decor's opening.** Each call scales its input to $t$ and partitions it like a Beaver multiplication: the compute
parties open $t - r$ for a mask $r$ drawn uniformly over the whole modulus (field or ring) from the pairwise PRG
streams. The opening is uniform and independent of the input. Every call draws a fresh mask: two openings against the
same mask would reveal the difference of the two inputs.

**The dealer's role.** CP0 knows $r$: it deals the shares of $h(r)$ that the decomposition combines, and an XOR sharing
of $r$'s fractional bits for the carry. As with Beaver triples, CP0 together with any one compute party can recover $t$.
Decor adds no trust beyond the existing [dealer assumption](#trusted-dealer-cp0).

**Bitwise sharing.** The carry, the ring sign test and the ring normalizer XOR-share bits among the compute parties,
packed into registers. Every mask is uniform over all bits of its register. What they open is masked uniformly over
the opened width: the XOR-masked operands of AND gates, Beaver partitions, the sign test's $-a + r \bmod 2^{\text{bits}}$,
the normalizer's $X - r \bmod 2^K$ and its masked one-hot bits, and each bit-to-arithmetic conversion's masked bit.

**Public checks.** Before any communication, every party checks the function, interval, degree and tolerance
(admissibility). The checks read public parameters only, so a refusal reveals nothing about the data.

**Inputs outside the interval** give wrong results (for example, $f(x - kp)$ for Decor's bounded functions), not a leak.
Only debug builds (`DEBUG = True`) open inputs, to warn about values outside the interval.

## Noise and masks in the MHE protocols

What each protocol reveals to the party that combines the shares (the hub):

- **Collective decryption.** Each compute party sends the hub $s_i \cdot c_1 + e_i$, with Gaussian noise $e_i$ of standard deviation `SEQURE_SIGMA_SMUDGE` (default $2^{10}$; values below 3.2 are rejected). Without noise, the hub could compute every key share as $s_i = \text{share} / c_1$; with it, recovering $s_i$ is a ring-LWE problem. The noise does not *flood* the ciphertext noise statistically, so a decrypted result still carries some information about the ciphertext's own noise (the IND-CPA-D setting). Flooding needs a standard deviation of about $2^{20}$ or more together with a larger scale, because the decryption error grows in proportion to it: the measured maximum error at the default scale $2^{34}$ is about $5 \cdot 10^{-5}$ at $2^{10}$, $5 \cdot 10^{-4}$ at $2^{14}$, and $3 \cdot 10^{-2}$ at $2^{20}$.
- **Decryption to the holder** (the default of `MPCMHE.decrypt` and `Ciphertensor.reveal`, behind `MPP.reveal_local`). Each compute party decrypts its own ciphertexts: it sends every other compute party the $c_1$ polynomial of each (never $c_0$) and gets back from each a decryption share $s_i \cdot c_1 + e_i$, smudged as above and sent to it alone; it adds its own $s \cdot c_1$ and $c_0$. The holder's view is the hub's view in a collective decryption (the same smudging guarantee). A party that is not the holder sees only $c_1$ polynomials, which are computed from ciphertexts and reveal nothing about the plaintexts under RLWE, and no share of anyone's decryption.
- **HE → MPC switching** (encryption to shares). Every compute party except the hub sends the hub its decryption share masked by a uniformly random polynomial, and keeps the negated mask. The hub decrypts the plaintext's coefficients under the others' masks, and each party decodes its coefficient share exactly into fixed-point shares (with the [exact encoder](../deep-dive-lattiseq/ckks-operations.md#exact-encoding)).
    - **Masks:** $\lambda + \log_2 N + \lceil \log_2 \Delta \rceil + k - f$ bits, where $\lambda$ = `MHE_MPC_SWITCH_LAMBDA` (128), $k$ and $f$ are the fixed-point total and fractional bits, and $N$ is the ring degree. That is 208 bits for $N = 2^{14}$ at scale $2^{34}$.
    - **Hiding:** for slot values in the fixed-point range ($|v| < 2^{k-f-1}$), the hub's view is within statistical distance $2^{-128}$ of one that depends on neither the plaintext nor its noise.
    - **Level:** the masks need a ciphertext modulus about two bits wider than themselves: level 5 of `PN14QP438` with three parties. Ciphertexts below that level are bootstrapped first; those above are dropped to it, which also makes the switch cheaper.
- **MPC → HE switching** (shares to encryption). Only the hub sees the opened value $x - R$, where $R$ is the sum of the compute parties' masks, and the parties then encrypt their shares jointly. Each party exactly encodes its share and masks it with $s_i \cdot a$ for a common random polynomial $a$; the sum is a fresh encryption under the collective key.
    - **Masks:** as large as keeps $x - R$ from wrapping around the MPC modulus, $\lceil \log_2 p \rceil - 2 - \lceil \log_2 (\text{compute parties}) \rceil$ bits. That is 171 bits with 192-bit shares and three parties.
    - **Hiding:** a $k$-bit value is hidden up to a statistical distance of $2^{k - 1} / 2^{\text{mask bits}}$. That is $2^{-108}$ with 192-bit shares, $2^{-61}$ with 128-bit shares and $2^{-121}$ with 256-bit shares.
- **Collective bootstrapping (refresh).** Partial decryptions are masked with random integers of $\lambda + \log_2 \Delta$ bits in the coefficient domain ($\lambda = 128$), which hide the plaintext and its noise statistically for slot values up to about 1.

Values beyond the fixed-point range are hidden less well, in proportion to their magnitude.

## What each reveal discloses

A reveal discloses its result to the parties named below and nothing else, with the guarantees above (the smudging of decryption shares, the masks of the switches). CP0 learns none of these results: it gets placeholders.

| Operation | Who learns what |
|---|---|
| `Sharetensor.reveal` | Every compute party: the value |
| `MPCMHE.decrypt(x)`, `Ciphertensor.reveal(mpc)` (default) | Each compute party: the plaintexts of its own ciphertexts |
| `MPCMHE.decrypt(x, p)` | Every compute party: the plaintexts of compute party `p`'s ciphertexts |
| `MPCMHE.decrypt(x, -1)` | Every compute party: the plaintext of ciphertexts that are the same at all of them (an MPA aggregate) |
| `MPP.reveal()`, `MPU.reveal()` (partition mode) | Every compute party: the whole matrix (all partitions) |
| `MPP.reveal_local()` | Each compute party: its own partition, its semi-encrypted blocks included |
| `MPA.reveal()`, `MPU.reveal()` (additive mode) | Every compute party: the sum of the shares. Shares in the clear go out under pairwise masks that cancel in the sum: each sent vector is uniform to a party that lacks one of its masks, so any coalition learns only the sum of the other parties' shares. With two compute parties, the sum and one's own share give the other's share: that is inherent to revealing a sum |
| `MPA.reveal_local()`, `MPU.reveal_local()` (additive mode) | Each compute party: its own share (in the clear, or decrypted to it alone). An MPA that holds only its aggregate, the same ciphertexts at every party, reveals the sum to every compute party |

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
