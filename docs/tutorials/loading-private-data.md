# Loading Private Data

The examples in `examples/` use **synthetic data shared from a trusted dealer** — all parties generate the same random values from a shared seed, which is convenient for quick tests but does not reflect how real deployments handle private data.

In production, **each party holds its own data on disk** and brings it into the secure computation via `MPU.collective_load`.  This tutorial explains how.

---

## The two data-loading paradigms

| | Test / synthetic | Production / file-based |
|---|---|---|
| **How** | `mpc.randomness.seed_switch(-1)` + manual slicing | `MPU.collective_load(mpc, path, rows, cols, binary, collective_type=...)` |
| **Who generates data?** | Every party generates the same array from a shared seed | Each party reads its own private file |
| **Trusted dealer (CP0)** | Holds a zero-filled slice | Receives metadata only (row counts); holds zeros |
| **When to use** | Unit tests, quick experiments | Real deployments with private data |

---

## Preparing data files

`collective_load` reads one ndarray file per party.  The `binary` flag controls the file format:

- **`binary=True`** — raw bytes (flat memory dump, no headers or delimiters).
- **`binary=False`** — text format (whitespace-delimited values).

To write a binary file from within Sequre:

```python
from numpy.create import array
from sequre.utils.io import write_ndarray
from sequre.utils.utils import __rand_mat

# Example: 4 rows × 4 columns of random integers
data = array(__rand_mat([4, 4], 100, TP=int))

with open("my_data.bin", "wb") as f:
    write_ndarray(f, data, binary=True)
```

When `binary=True`, each file is a flat dump of the ndarray's memory — no headers, no delimiters, just raw element bytes.

!!! tip
    In a distributed deployment every party naturally has its own filesystem, so file paths can be identical across machines (e.g. `/data/patients.bin`).  In **local mode** (all parties on one machine) paths must differ — parameterise by party ID: `f"data/cp{mpc.pid}.bin"`.

---

## `MPU.collective_load`

`MPU.collective_load` is the main API for loading private data from files.  The `collective_type` argument controls how data is distributed:

- **`"partition"`** — horizontal partitioning (backed by `MPP`).  Each party contributes its own rows; the global matrix is the vertical concatenation of all parties' data.
- **`"additive"`** — additive sharing (backed by `MPA`).  Each party holds one additive share; the logical value is the sum of all shares.

### Partition mode

Each compute party reads its own rows.  Parties exchange row counts so everyone knows the partition layout.

```python
from sequre.types.multiparty_union import MPU

data_path = f"data/credit_scoring_cp{mpc.pid}.bin"

X = MPU.collective_load(
    mpc, data_path,
    rows=rows_per_party,   # number of rows *this* party contributes
    cols=features,          # must be the same for all parties
    binary=True,
    dtype=int,
    collective_type="partition")
```

Best for: horizontally partitioned datasets where each party owns a disjoint subset of records (e.g. each hospital has its own patients).

### Additive mode

Each compute party reads its own share from file.

```python
X = MPU.collective_load(
    mpc, data_path,
    rows=rows_per_party,
    cols=features,
    binary=True,
    dtype=int,
    collective_type="additive")
```

Best for: pre-shared additive data or scenarios where the logical value is the sum of contributions from all parties.

### Signature

```
MPU.collective_load[dtype](
    mpc:             MPCEnv,
    data_path:       str,
    rows:            int,
    cols:            int,
    binary:          bool,
    collective_type: Static[str]     # "partition" or "additive"
) -> MPU[Tuple[int, int], dtype]
```

---

## Lower-level APIs

For cases where you want to work with the underlying types directly, `collective_load` is also available on the lower-level multiparty types and on `Sharetensor`:

| Type | Use case |
|---|---|
| `MPP.collective_load(mpc, path, rows, cols, binary, dtype=...)` | Horizontal partitioning (MHE) — returns `MPP` |
| `MPA.collective_load(mpc, path, rows, cols, binary, dtype=...)` | Additive sharing (MHE) — returns `MPA` |
| `Sharetensor.collective_load(mpc, path, rows, cols, binary, dtype=...)` | Additive secret sharing (MPC) — returns `Sharetensor` |

---

## Party roles

| Party | Role | Data |
|---|---|---|
| **CP0** (pid 0) | Trusted dealer | Holds zeros — does not contribute private data |
| **CP1 … CPN** (pid > 0) | Compute parties | Each holds its own private rows loaded from a file |

The trusted dealer participates in the cryptographic protocol (key generation, share distribution) but never sees any party's raw input.

---

## Full working example

See [`examples/collective_load.codon`](https://github.com/0xTCG/sequre/blob/develop/examples/collective_load.codon) for a complete, runnable file that demonstrates both modes:

- **Credit scoring (partition mode):** loads customer records via `MPU.collective_load(..., collective_type="partition")`, trains a neural network on the encrypted partitions.
- **Linear regression (MPC):** loads patient records via `Sharetensor.collective_load`, trains a regression model on additive secret shares.

Run it with:

```bash
sequre examples/collective_load.codon --local
```

---

## When to use which mode?

| Mode | Backed by | Best for |
|---|---|---|
| `"partition"` | `MPP` — horizontally partitioned | Each party owns distinct records; batched linear algebra, neural networks, large-scale data |
| `"additive"` | `MPA` — additive shares | Pre-shared data; scenarios where the logical value is the sum of all parties' contributions |

You can also switch between MHE and MPC mid-computation using `via_mpc` — see [MPC ↔ MHE Switching](../user-guide/switching.md).
