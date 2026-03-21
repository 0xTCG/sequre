# Multiparty Types

_Defined in `stdlib/sequre/types/multiparty_union.codon`, `multiparty_partition.codon`, `multiparty_aggregate.codon`_

These three types model how data is distributed across parties in a multiparty computation. They are the primary abstractions for Shechi's high-level layer.

| Type | Description | Detailed Reference |
|---|---|---|
| **MPU** | Multiparty Union — highest-level distributed type. A union of MPP and MPA that selects the appropriate representation internally. | [MPU Reference](mpu.md) |
| **MPP** | Multiparty Partition — each party holds a contiguous block of rows (horizontal partitioning). | [MPP Reference](mpp.md) |
| **MPA** | Multiparty Aggregate — each party holds an additive share of the global value, optionally encrypted. | [MPA Reference](mpa.md) |

## Quick start

```python
from sequre.types.multiparty_union import MPU

# Horizontal partition: each party owns a subset of rows
mpu = MPU(mpc, my_local_rows, "partition")

# Additive sharing: each party holds an additive share
mpu = MPU(mpc, my_share, "additive")

# All standard operators work
result = mpu_a @ mpu_b
mask = mpu > 0.5
```

## Choosing between partition and additive

| Mode | Internal type | Best for |
|---|---|---|
| `"partition"` | MPP | Data naturally split by rows (e.g., each hospital holds patients) |
| `"additive"` | MPA | Data that needs to be secret-shared across all parties |

## See also

- [SMC ↔ MHE Switching](../user-guide/switching.md) — How `via_mpc` enables switching between protocols mid-computation
