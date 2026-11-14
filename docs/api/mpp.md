# MPP — Multiparty Partition

_Defined in `stdlib/sequre/types/multiparty_partition.codon`_

`MPP[S, dtype]` represents data that is **horizontally partitioned** across parties. Each party holds a contiguous block of rows of the global matrix.

---

## Fields

| Field | Description |
|---|---|
| `_mpc` | Reference to the MPC instance |
| `_ratios` | List of per-party row counts |
| `_local_data` | This party's local plaintext rows |
| `_encryption_unified` | Encrypted version of the local data (`Ciphertensor[Ciphertext]`) |

## Construction

```python
from sequre.types.multiparty_partition import MPP

# From local data (each party provides its own rows)
mpp = MPP(mpc, local_rows)

# From a Ciphertensor (each party provides encrypted local data)
mpp = MPP(mpc, local_ciphertensor)
```

!!! note
    Party 0 (trusted dealer) holds a zero-filled placeholder. Actual data resides on parties 1..N.

The compute parties' partitions may differ in size, and a party may hold no rows at all (a ratio of 0). Call every operation at every party, the trusted dealer and parties with empty partitions included: the parties take the same sequence of collective steps whatever their partitions hold.

The compute parties may also hold their partitions differently: some in the clear and others encrypted (a *mixed cohort*, for example after `encrypt()` at some of them only), or some semi-encrypted with different blocks (the result of `A @ B.T` in the clear). Element-wise operations, powers, matrix products and sums handle any mix. Element-wise operations, powers, products with a plaintext matrix and sums within the partitions keep each party's own representation: a partition in the clear gives a result in the clear, an encrypted one an encrypted result. Sums across the partitions give a row the hub holds encrypted (see below). Products of two secret MPPs (`A.T @ B`, `A @ B.T`) encrypt the partitions in the clear of a mixed factor for the product, by public-key encryption; their results are encrypted. A product refreshes its encrypted operands, and each refresh is a council among the compute parties, so operations that may refresh (multiplications, powers, products) first agree on how every party holds its partition, in one round among the compute parties; a party that holds no encryption then runs the others' HE steps on placeholders. Element-wise sums and differences (`+`, `-`) of partitions of equal size need no round.

## One-dimensional MPPs and sums

A 1-dim MPP is one of two things, told apart by its lazy transpose:

- **Not transposed: a row held by one compute party** (the holder). Its ratios are one-hot at the holder, whose partition is the row; the other compute parties hold nothing, and CP0 a placeholder of the row's length. `X[i]` gives one.
- **Transposed: a column split over the compute parties.** Its ratios are each party's length (0 for an empty partition), and CP0 holds a placeholder of the total length. `X.T[j]` gives one.

A vector of length 1 held by one party with one-hot ratios reads the same either way. `MPP(mpc, vector)`, which takes each party's vector, builds an untransposed 1-dim MPP whose ratios are the lengths: sums read it as a column, as `Sharetensor.to_mpp` reads ratios that sum to more than 1.

Sums follow from where they add up. Of a 2-dim MPP `X` (its partitions are rows of the matrix as held):

| Sum | Adds up | Result |
|---|---|---|
| `X.sum(1)`, `X.T.sum(0)` | within each partition (each row as held) | a column split with `X`'s ratios: each party's sums of its own rows, in its own representation (in the clear, or encrypted; a semi-encrypted partition's sums encrypted) |
| `X.sum(0)`, `X.T.sum(1)` | across the partitions (each column as held) | a row held by the hub (CP1), encrypted: each party's partial sums, encrypted, go to the hub alone, which adds them |

A held row's sum stays with its holder (a length-1 row there, in the holder's representation); a column's sum is a column of length 1 at the hub, encrypted, which `broadcast` spreads over a column encrypted at every compute party with more than one element in each partition, and whose partitions' lengths are all multiples of the slots or none (`v - v.sum(0)`). `dot(axis)` is `(x * x).sum(axis)`.

The compute parties agree on how they hold their partitions in one round, and each sums its own as it holds it; a sum along the ciphertext rows of an encryption refreshes once per row, so a party whose sum takes fewer refreshes (in the clear, an empty partition, fewer rows) makes the rest on placeholders, and every compute party takes the same collective steps. A sum that some compute party cannot compute raises `NotImplementedError` at every compute party; CP0 has returned its placeholder by then, so a caller that catches the error must not fall back to a step that involves CP0. These are: along the rows of a diagonal storage; a skinny diagonal storage whose rows span several ciphertexts that it cannot fold; a semi-encrypted partition whose blocks are not column-wise, summed across the partitions; as many storage rows to reduce as there are slots.

## Factory methods

| Method | Description |
|---|---|
| `MPP.enc(mpc, data, ratios=None)` | Create encrypted partition with optional ratios |
| `MPP.collective_load(mpc, path, rows, cols, binary)` | Load partitioned matrix from files across parties |
| `MPP.rand(shape, distribution, mpc)` | Random partition |
| `MPP.like(other, arr)` | Create MPP matching another's structure |
| `MPP.uniform_ratios(size, n_parties)` | Generate evenly distributed partition ratios |
| `MPP.apply_ratios(ratios, target_size, approximate)` | Scale ratios proportionally |

## Properties and state queries

| Property / Method | Type | Description |
|---|---|---|
| `.shape` | `S` | Current shape (transposition-aware) |
| `.cohort_shape` | `S` | Collective shape across all parties |
| `.shape_local` | `S` | Alias for shape |
| `.partition_shape` | `S` / `list[int]` | Raw partition shape |
| `.ndim` | `int` | Number of dimensions |
| `.T` | `MPP` | Lazy transpose |
| `.I` | `MPP` | Identity matrix partition |
| `.local` | `ndarray` | Local plaintext data (requires `is_local`) |
| `.modulus` | `mpc_uint` | MPC modulus |
| `.ratios` | `list[int]` | Partition size ratios |
| `.has_partial_encryption()` | `bool` | Left/right encrypted portions exist? |
| `.has_unified_encryption()` | `bool` | Full partition encrypted? |
| `.has_local_data()` | `bool` | Plaintext data present? |
| `.is_semi_encrypted()` | `bool` | Mixed plaintext/ciphertext? |
| `.is_full_encrypted()` | `bool` | Fully encrypted? |
| `.is_local()` | `bool` | Plaintext only? |
| `.is_empty()` | `bool` | No data present? |
| `.cohort_is_aligned()` | `bool` | All parties have equal partition size? |

## Operators

| Operator | Description |
|---|---|
| `a + b` | Secure addition |
| `a - b` | Secure subtraction |
| `a * b` | Secure multiplication |
| `a @ b` | Secure matrix multiplication |
| `a / b` | Division by scalar |
| `a ** n` | Integer exponentiation |
| `-a` | Negation |
| `a > b` | Greater-than a public value: local at each compute party if all of them hold their partitions in the clear, otherwise the secure comparison through [MPC switching](../user-guide/switching.md#comparisons) |
| `a < b` | Less-than a public value, likewise |
| `a[i]`, `a[i] = v` | Indexing and slicing |
| `bool(a)` | Non-empty check |
| `len(a)` | Length of first dimension |

## Reveal and encryption

| Method | Description |
|---|---|
| `.encrypt()` | Encrypt local plaintext to ciphertext: row-wise, or column-wise when the MPP is lazily transposed (the same layout at every party, whatever the partitions' shapes) |
| `.reveal()` | Reveal with secure aggregation to all parties |
| `.reveal_local()` | This compute party's own partition, decrypted to it alone: no other party learns it. Call it at every compute party, since each takes part in the decryption whatever it holds (in the clear, encrypted, empty). A semi-encrypted partition's blocks (from `A @ B.T`) are its holder's. CP0 gets its placeholder |
| `.level()` | Minimum encryption level |
| `.unify_encryption(transposed=-1)` | Cache this party's partition as one encryption: a semi-encrypted partition's blocks and block in the clear concatenated, in the blocks' layout; a 2-dim partition in the clear row-wise as held, or column-wise for `transposed=1` (a layout the caller agreed on with the other parties) |

## Distributed operations

| Method | Description |
|---|---|
| `.collect()` | Collect encryptions from all parties |
| `.collect_and_execute_at(target_pid, op)` | Collective operation executed at target party |
| `.aggregate()` | Aggregate to aligned uniform partition |
| `.aggregate_at(target_pid)` | Aggregate all shares at specific party |
| `.join()` | The whole cohort as one encryption, at every compute party: for a 2-dim MPP the parties agree on their partitions in one round, and partitions held in the clear are encrypted in the layout of the encrypted ones (row-wise as held if none is encrypted), so that a transposed MPP encrypted column-wise at some parties joins with the others. Encrypted layouts that differ, and diagonal storages, raise at every compute party (CP0 has returned by then). A vector's parts join with their transposed flags cleared |
| `.join_at(target_pid)` | Join at specific target party |
| `.broadcast_from(source_pid)` | Broadcast partition from source party |
| `.broadcast(target)` | Spread a transposed vector of length 1, encrypted (such as a column's sum), over a transposed MPP encrypted at every compute party whose partitions all hold more than one element, their lengths all multiples of the slots or none. An element-wise operand whose ratios sum to 1 goes here; one it cannot take (a held row, such as a column sum; a non-transposed target, a partition of fewer than two elements, lengths that are multiples of the slots at some partitions only; either of them held in the clear at some party) raises `NotImplementedError` at every party (the state takes one round with every party, CP0 included), and an element-wise operation between a 2-dim and a 1-dim MPP does not compile |
| `.align_ratios(ratios, force)` | Realign partition ratios |
| `.sum(axis)` | Sum along axis: a column split within the partitions, a row at the hub across them (see [One-dimensional MPPs and sums](#one-dimensional-mpps-and-sums)). Call it at every party |
| `.dot(axis)` | `(x * x).sum(axis)` |

## Shape manipulation

| Method | Description |
|---|---|
| `.expand_dims(axis=0)` | Insert dimension |
| `.extend(other)` | Concatenate partitions |
| `.pad_with_value(val, size, axis, ...)` | Append `size` columns of `val` to every party's rows (axis 1, or axis 0 of a transposed MPP), in each partition's representation |
| `.erase_element(index)` | Remove element by index |
| `.filter(mask)` | Boolean mask filtering |
| `.cohort_filter(mask)` | Filter across cohort with adjusted ratios |
| `.hstack(other)` | Horizontal concatenation |
| `.local_broadcast(target_shape)` | Broadcast single element to shape |
| `.replicate(new_size)` | Copy with size expansion |
| `.actual_transpose()` | Materialize actual transpose (vs. lazy `.T`) |
| `.actual_itranspose()` | In-place actual transpose |
| `.getitem_local(index)` | Local indexing on plaintext data |
| `.slice_local(idx, ...)` | Local slicing on transposed encrypted data |
| `.rotate_local(i, ...)` | Local rotation on transposed data |

## Protocol switching

| Method | Description |
|---|---|
| `.via_mpc(fn, *args)` | Execute function through the [MPC layer](../user-guide/switching.md) (converts to Sharetensor, runs fn, converts back) |
| `.sign(*args)` | Secure sign function |

!!! info
    Comparisons (`>`, `<`) and certain `matmul` cases automatically use `via_mpc` under the hood, and a product with a plaintext matrix may under `AllowMPCSwitch`. See the [MPC ↔ MHE Switching](../user-guide/switching.md) page.

## Conversion and creation

| Method | Description |
|---|---|
| `.astype(T)` | Cast to different dtype |
| `.to_fp()` | Convert to floating point |
| `.copy()` | Deep copy all components |
| `.set(other)` | Copy state from another MPP |
| `.zeros(shape)` / `.zeros()` | Zero tensor |
| `.ones(...)` | Ones tensor |
| `.rand(distribution, ...)` | Random partition |
| `.get_matmul_cost(other)` | Estimate matmul cost |
| `.get_relative_indices()` | Local partition start/end indices |
