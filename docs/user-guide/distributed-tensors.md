# Distributed Tensors (MPU)

`MPU` (Multiparty Union) is Sequre/Shechi's highest-level abstraction for distributed secure computation. It lets you write code that looks like standard tensor arithmetic while the framework handles encryption, communication, and protocol selection.

## When to use MPU

Use `MPU` when your data is **distributed across parties** — for example, multiple hospitals each hold patient records and want to jointly train a model without sharing raw data.

## Construction

```python
from sequre.types.multiparty_union import MPU

# Horizontal partition: each party owns some rows
mpu = MPU(mpc, my_local_rows, "partition")

# Additive sharing: each party holds one additive share
mpu = MPU(mpc, my_share, "additive")
```

### Partition mode (`"partition"` → MPP internally)

Each compute party (CP1..N) provides its own block of rows. Party 0 (trusted dealer) provides a zero placeholder. The `MPP` tracks per-party row counts (`_ratios`) so the global shape is known.

```
Global matrix (100 rows):
  CP1 holds rows  0..49   (50 rows)
  CP2 holds rows 50..99   (50 rows)
  CP0 holds zeros(100, cols)  ← placeholder
```

### Additive mode (`"additive"` → MPA internally)

Each party holds a share; the sum of all shares equals the original value.

## Operators

All standard operators work on MPU:

```python
c = a + b       # Addition
c = a @ b       # Matrix multiplication
mask = mpu > 0.5  # Comparison
row = mpu[i]    # Indexing
```

The framework dispatches to the correct protocol (MPC or MHE) based on internal state.

## Under the hood: MPP and MPA

### MPP (Multiparty Partition)

`MPP` stores data horizontally partitioned across parties. Each party has:

- `_local_data`: its own rows as an ndarray
- `_encryption_unified`: encrypted version as a `Ciphertensor[Ciphertext]`
- `_ratios`: list of row counts per party

Operations that require the full dataset transparently encrypt local data and run collective HE operations.

### MPA (Multiparty Aggregate)

`MPA` stores additive shares with optional encryption. Key methods:

- `.via_mpc(fn)` — convert to `Sharetensor`, apply a function via MPC, and convert back
- `.enc()` — encrypt the plaintext share into a `Ciphertensor`

## Practical example: DTI

From `applications/dti.codon` — each party holds a horizontal slice of training data:

```python
@sequre
def dti_mhe_protocol(mpc):
    X_mpu = MPU(mpc, X_partition, "partition")
    y_mpu = MPU(mpc, y_partition, "partition")

    model = Sequential(...)
    model.fit(mpc, X_mpu, y_mpu, epochs=100)
    return model.predict(mpc, X_mpu)
```

The `Sequential` model handles the collective HE operations internally.

## Next steps

- **[Secure Local Tensors (Ciphertensor)](secure-local-tensors.md)** — The layer beneath MPU.
- **[MPU / MPP / MPA API](../api/multiparty-types.md)** — Complete reference.
- **[Core MHE module](../deep-dive-shechi/core-mhe.md)** — How collective operations work.
