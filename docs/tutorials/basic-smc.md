# Basic SMC with Sequre

This tutorial walks through writing secure multiparty computations using Sequre's additive secret sharing layer.

## Core idea: additive secret sharing

In Sequre's baseline MPC protocol, a secret value \(x\) is split into \(n\) additive shares:

\[
x = \sum_{i=1}^{n} x_i \pmod{p}
\]

Each party \(i\) holds only \(x_i\). No individual share reveals anything about \(x\).

**Addition is free.** Each party locally computes \(x_i + y_i\) to obtain a share of \(x + y\).

**Multiplication requires communication.** Sequre uses Beaver triples — pre-shared random correlations \((a, b, c)\) where \(c = a \cdot b\). The protocol requires one round of communication.

## The `@sequre` decorator

Any function annotated with `@sequre` has its operators on secure types automatically rewritten by the compiler plugin:

```python
from sequre import sequre, Sharetensor as Stensor


@sequre
def secure_sum_of_products(mpc, a, b, c):
    return a * b + b * c + a * c
```

Under the hood, `a * b` becomes a call to `secure_mul` which runs the Beaver triple protocol. Addition stays local.

## Writing a complete program

### Local mode (single machine)

```python
from sequre import local, sequre, Sharetensor as Stensor


@sequre
def private_dot(mpc, x, y):
    return x.dot(mpc, y, axis=0)


@local
def main(mpc):
    x = Stensor.enc(mpc, [1, 2, 3, 4, 5])
    y = Stensor.enc(mpc, [5, 4, 3, 2, 1])
    result = private_dot(mpc, x, y)
    print(f"CP{mpc.pid}: dot product = {result.reveal(mpc)}")


main()
```

Run:
```bash
./bin/sequre script.codon
```

### Online mode (across machines)

```python
from sequre import mpc, sequre, Sharetensor as Stensor


@sequre
def private_dot(mpc, x, y):
    return x.dot(mpc, y, axis=0)


mpc = mpc()
x = Stensor.enc(mpc, [1, 2, 3, 4, 5])
y = Stensor.enc(mpc, [5, 4, 3, 2, 1])
result = private_dot(mpc, x, y)
print(f"CP{mpc.pid}: dot product = {result.reveal(mpc)}")
```

Run at each party:
```bash
SEQURE_CP_IPS=192.168.0.1,192.168.0.2,192.168.0.3 ./bin/sequre script.codon <pid>
```

## Party model

Sequre uses a **trusted dealer** model with at least 3 parties:

| Party | Role |
|---|---|
| CP0 | Trusted dealer — generates Beaver triples \& random masks |
| CP1..N | Compute parties — hold shares and participate in protocols |

!!! note
    In local mode (`@local`), all parties are forked on the same machine using UNIX sockets. In online mode, they communicate via TCP with TLS.

## Fixed-point arithmetic

For computation on real numbers, Sequre uses fixed-point encoding:

```python
# Values are converted to fixed-point internally
# After multiplication, truncate to maintain precision
result = (a * b).trunc(mpc.fp)
```

Precision is controlled by `MPC_NBIT_K` (total bits), `MPC_NBIT_F` (fractional bits).

## Comparisons

Sequre supports secure comparisons through bit decomposition:

```python
@sequre
def find_greater(mpc, x, threshold):
    return x > threshold  # Returns secret-shared 0/1 values
```

These are more expensive than arithmetic but fully supported.

## Next steps

- **[Transitioning to MHE](transition-mhe.md)** — When to prefer homomorphic encryption over secret sharing.
- **[Sharetensor API](../api/sharetensor.md)** — Complete reference.
- **[MPCEnv API](../api/mpcenv.md)** — Sub-modules and methods.
