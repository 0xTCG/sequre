# Secure Branching Without `if`

Sequre deliberately disallows branching (`if`) on secret data. I.e. the following will not work whenever `x` or `y` are encrypted.

```python
if x > y:
    z = x
else:
    z = y
```

Otherwise, both branches would need to be evaluated not to leak whether `x > y`.

To implement `max`, `min`, `clip`, or any other conditional logic on secrets, the branching is replaced with arithmetic.

## How it works

The idea is simple: compute the comparison as a secret-shared 0-or-1 value, then use it as an arithmetic mask to select between two outcomes — without ever branching.

Here is the actual `maximum` from Sequre's stdlib ([stdlib/sequre/stdlib/builtin.codon](../../stdlib/sequre/stdlib/builtin.codon)):

```python
@sequre
def maximum(mpc, x, y):
    mask = ((x - y) > 0).astype(float)
    return x * mask - y * (mask - 1)
```

Walk through it: `(x - y) > 0` produces a secret-shared bit — `1` if `x > y`, `0` otherwise. Casting it to float gives a mask that is either `1.0` or `0.0`. Then:

- If `mask = 1`: result is `x * 1 - y * 0 = x`. Correct — `x` was bigger.
- If `mask = 0`: result is `x * 0 - y * (-1) = y`. Correct — `y` was bigger.

No branch taken, no information leaked. Both multiplication paths are always executed.

`minimum` is similar:

```python
@sequre
def minimum(mpc, x, y):
    mask = ((y - x) > 0).astype(float)
    return x * mask - y * (mask - 1)
```

## More examples from the codebase

### Clipping

`clip` needs to handle two boundaries at once — a low threshold and a high threshold. It builds two masks and combines them:

```python
@sequre
def clip(mpc, x, low, high):
    low_mask = (x < low).astype(float)
    high_mask = (x > high).astype(float)
    return x * (1 - (low_mask + high_mask)) + low_mask * low + high_mask * high
```

If `x` is below `low`, the `low_mask` fires and the result is `low`. If it's above `high`, the `high_mask` fires. Otherwise both masks are zero and you get `x` unchanged.

### Absolute value

```python
@sequre
def abs(mpc, x):
    return x * (((x > 0) * 2) - 1)
```

This computes `sign(x)` as `+1` or `-1` and multiplies. If `x > 0`, the factor is `(1*2) - 1 = 1`. If `x <= 0`, it's `(0*2) - 1 = -1`.

### Argmax

`argmax` is more involved — it walks through a vector, keeping a running maximum and the index that produced it. At each step it uses `max` to do a branchless comparison and update:

```python
@sequre
def argmax(mpc, x):
    arg, maximum = Sharetensor(0, x.modulus), x[0]

    for i in range(1, len(x)):
        new_maximum = max(mpc, maximum, x[i])
        arg = max(mpc, arg, (new_maximum > maximum) * i)
        maximum = new_maximum

    return arg, maximum
```

Notice `(new_maximum > maximum) * i` — this produces `i` if the new element was bigger, or `0` otherwise. Then `max(mpc, arg, ...)` picks the larger of the current argmax and this candidate. All branchless.

## What to keep in mind

**Both sides always execute.** Unlike a plaintext `if`, the expensive branch cannot be skipped. If one path involves a heavy computation, the cost is paid regardless of the condition.

**Comparisons are not cheap.** Under the hood, `x > y` on secret-shared data involves bit decomposition — significantly more expensive than addition or multiplication. If an algorithm does many comparisons, that will dominate the cost. Restructuring to minimize the number of comparison operations is advisable.

**There's also oblivious array access.** When indexing into an array at a secret position (not just pick between two values), Sequre provides `oblivious_get` in [stdlib/sequre/mpc/collections.codon](../../stdlib/sequre/mpc/collections.codon). It uses a demultiplexer built from bit decomposition ([stdlib/sequre/mpc/boolean.codon](../../stdlib/sequre/mpc/boolean.codon)) to read a public array at a secret index without revealing which element was accessed. This scales as $O(2^{\text{bits}})$, so best to keep the key bit length small.

## Existing built-ins

Before writing custom mask logic, check if Sequre already provides what is needed — `maximum`, `minimum`, `clip`, `abs`, `sign`, and `argmax` are all in `stdlib/sequre/stdlib/builtin.codon`.

## Next steps

- [Basic MPC Computation](basic-mpc.md) — How secure arithmetic and comparisons work underneath.
- [Secure Stdlib API](../api/stdlib.md) — Full reference for the built-in secure functions.
- [MPC ↔ MHE Protocol Switching](../user-guide/switching.md) — How comparisons on encrypted (MHE) data switch to MPC automatically.
