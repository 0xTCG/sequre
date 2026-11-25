# Decorators & Attributes

_Defined in `stdlib/sequre/attributes.codon`, `stdlib/sequre/decorators.codon`, and `stdlib/sequre/runtime.codon`_

Sequre uses compile-time attributes (processed by the Sequre IR plugin) and runtime decorators to control how functions are transformed, optimized, and executed.

---

## Compiler attributes

These decorators are processed by the Sequre compiler plugin during IR transformation passes. They do not exist at runtime — they instruct the compiler to rewrite annotated functions.

### `@sequre`

The primary attribute. Marks a function for Sequre's IR rewriting pipeline. The compiler:

1. **Expressiveness transformations** — rewrites operator overloads (e.g., `a > b` on encrypted types dispatches to secure comparison)
2. **MPC optimizations** (`@mpc_poly_opt`) — polynomial-evaluation optimizations for MPC operations
3. **MHE optimizations** (`@mhe_mat_opt`, `@mhe_cipher_opt`, `@mhe_enc_opt`) — opt-in optimizations; `@mhe_cipher_opt` factors and reorders a function's arithmetic on Sharetensors, and on a single Ciphertensor or MP tensor among public operands; `@mhe_enc_opt` is accepted and does nothing

```python
from sequre.attributes import sequre

@sequre
def my_secure_function(mpc, x, y):
    return x @ y + x * y  # operators are rewritten to secure versions
```

!!! note
    Every function that operates on Sequre types (`Sharetensor`, `Ciphertensor`, `MPU`, etc.) should be annotated with `@sequre`.

<!-- ### `@mpc_poly_opt`

Enables polynomial optimization for MPC operations. The compiler rewrites sequences of multiplications and additions into more efficient polynomial evaluation forms to minimize the number of Beaver-triple rounds.

### `@mhe_mat_opt`

Enables matrix-multiplication optimization for MHE. The compiler analyzes matrix operands and selects the best multiplication strategy (row-packed, column-packed, or diagonal-packed).

### `@debug`

Enables debug instrumentation. The compiler inserts additional logging, assertions, and timing instrumentation into the annotated function. -->

### `@mhe_cipher_opt`

An opt-in rewrite of a function's arithmetic: fewer secure products and additions for the same results. Put it under `@sequre`, on a function whose first argument is the MPC instance:

```python
from sequre import sequre, mhe_cipher_opt

@sequre
@mhe_cipher_opt
def f(mpc, a, b, c, d):
    return a * c + a * d + b * c + b * d  # on Sharetensors: (a + b) * (c + d), one secure product instead of four
```

In each top-level statement of the function it:

- **factors** an operand shared by terms of a sum: `a*c + a*d` becomes `a * (c + d)`;
- **combines public operands first** (ndarrays, numbers, `Ciphertensor[Plaintext]`): `a*b + a*c + a*d`, with `b`, `c` and `d` public, becomes `a * ((b + c) + d)`, one secure product;
- **folds a definition into its only use** when only plain assignments lie between them: `e = a*c + a*d`, `f = b*c + b*d`, `return e + f` is factored as one sum, and the assignments go (nothing else is ever removed);
- **orders chains of `@` of Sharetensors**: `x @ y @ z`, three or more Sharetensor matrices, runs in the order of fewest rows × inner × columns when that is strictly cheaper than the written order. Chains of other types run as written: their products' cost depends on layouts known only once the products exist.

On Sharetensors it rewrites any such expression. On Ciphertensors and MP tensors (`MPP`, `MPA`, `MPU`) it rewrites an expression only when it holds one of them, the other operands public: `a*c + a*d` with public `c` and `d` becomes `a * (c + d)`, but `a*c + a*d` of three Ciphertensors stays as written (below).

What it promises:

- The function gives the same results as without the decorator, up to fixed-point and CKKS rounding, with all of its effects in their order, at every party. That includes what Sequre's products do to their operands in place (refreshes, encryptions, aggregations): a Ciphertensor argument the written function would have bootstrapped comes back as high.
- A rewrite never costs more: fewer secure `@` products, then fewer secure-by-secure products, then fewer secure-by-public products, then fewer secure additions; no `@` product evaluated more often than written, and no secure value under more products (levels) than before. Anything it cannot prove so stays as written.
- It applies once per function, however many call sites the function has.

What it leaves as written:

- Expressions inside branches and loops, and expressions holding a call (other than `x.T` and `x @ y`), a ternary, an assignment expression or a list. The operands it plans are variables, `int` and `float` literals, field reads (`self.w`), `x.T` and `x @ y`, of type `int`, `float`, `ndarray`, `Sharetensor`, `Ciphertensor`, `MPP`, `MPA` or `MPU`.
- Pairings Sequre does not run: a `Ciphertensor[Plaintext]` meets only `Ciphertensor[Ciphertext]` operands, and secure operands of two different types are combined only as the function already combines them.
- Integer arithmetic that would wrap: two public integers (`int` values, ndarrays of ints) are never combined with each other, since their product or sum wraps at 64 bits where the secure arithmetic does not, unless both are literals and the result fits. An expression that already computes such a sum or product (`a * (n * m)`) stays as written.
- Constants out of range next to Sharetensors: Sequre encodes a public value added to or multiplied with a Sharetensor in 31 integer bits, so there only literals are combined, and only while the combined value stays within that range.
- Additions of a fixed-point and an integer Sharetensor, which Sequre refuses, and of a float to an integer Sharetensor, which it gets wrong. A rewritten expression runs only when each addition it makes adds two of a kind, and a float only to a fixed-point Sharetensor (checked at run time: a product is fixed point when a factor is, and a float makes it so); otherwise the expression runs as written.
- Expressions of two or more secure Ciphertensors or MP tensors (`x` and `x.T` count as two). Their storage is not in their types: a Ciphertensor is stored transposed (as `b.T` is) or diagonal, an MPP or an MPU of partitions transposed or split differently among the parties, an MPA aggregated or not. An element-wise operation of two tensors stored differently converts one to the other's storage (a Ciphertensor's through MPC), and their products refresh, encrypt or aggregate operands in place, so a new pairing of two of them could change what the function converts, bootstraps and returns. Choosing their storage is left to a layout optimization.
- Extra levels: an MPP or MPA times an int below 7 costs no level (Sequre adds), so such constants are not combined into a larger one.

!!! warning
    `@mhe_cipher_opt` assumes that every tensor in a decorated expression has the same shape (numbers broadcast). A function that adds or multiplies tensors of different shapes, such as a matrix and a row, must not use it.

Set `SEQURE_IR_OPT_DEBUG=1` at compile time to see each rewrite.

### `@mhe_enc_opt`

Accepted, and does nothing: a function under it compiles and runs as without it. It used to encrypt the public operand of each `Ciphertensor @ ndarray` product in an encoding chosen by a search; on today's kernels that was slower than the matmul routers' own choice everywhere it was measured (the routers pick, per product, the cheapest kernel and the encoding of the plaintext operand by cost).

---

## Runtime decorators

### `@flatten(idx)`

_Defined in `stdlib/sequre/decorators.codon`_

A runtime decorator that automatically flattens the first `idx` tensor arguments before calling the function, and reshapes the result back to the original shape. Useful for functions that operate on 1-D vectors but should accept matrices transparently.

```python
from sequre.decorators import flatten

@flatten(1)
def fp_div(mpc, a, b):
    # a and b are flattened to 1-D here
    ...
    # result is reshaped back to original shape of args[1]
```

### `@local`

_Defined in `stdlib/sequre/runtime.codon`_

A runtime decorator that forks the current process into `N` parties (using `fork()`), each running the decorated function as a separate MPC party with its own MPC instance. Used for local testing where all parties run on a single machine.

!!! warning
    The decorated function must be called **exactly once** per program. Calling it more than once will raise an error. This applies to `@local`, `@online`, and `@main` alike.

```python
from sequre.runtime import local

@local
def my_protocol(mpc):
    # Each forked process gets its own mpc with a unique pid
    X = MPU(mpc, local_data, "partition")
    result = X @ X.T
    print(f"CP{mpc.pid}: done")
```

Command-line flags (e.g., `--use-ring`, `--skip-mhe-setup`) are parsed from `sys.argv` and passed as control toggles.

### `@online`

_Defined in `stdlib/sequre/runtime.codon`_

A runtime decorator for distributed (multi-machine) execution. Wraps the `mpc()` lifecycle: parses the party ID from `sys.argv`, creates an MPC instance, calls the decorated function, and cleans up with `mpc.done()`.

!!! warning
    Must be called **exactly once** per program (same as `@local` and `@main`).

```python
from sequre.runtime import online

@online
def my_protocol(mpc):
    X = MPU(mpc, local_data, "partition")
    result = X @ X.T
    print(f"CP{mpc.pid}: done")

my_protocol()  # party ID is parsed from sys.argv
```

Run on each machine:
```bash
SEQURE_CP_IPS=192.168.0.1,192.168.0.2,192.168.0.3 sequre my_protocol.codon <pid>
```

### `@main`

_Defined in `stdlib/sequre/runtime.codon`_

A runtime decorator that lets the user control the execution mode via CLI. If `--local` is present in `sys.argv`, it runs the function via `@local` (forking parties on a single machine). Otherwise, it runs via `@online` (distributed execution).

!!! warning
    Must be called **exactly once** per program (same as `@local` and `@online`).

```python
from sequre.runtime import main

@main
def my_protocol(mpc):
    X = MPU(mpc, local_data, "partition")
    result = X @ X.T
    print(f"CP{mpc.pid}: done")

my_protocol()
```

```bash
# Local:
sequre my_protocol.codon --local

# Online (on each machine):
SEQURE_CP_IPS=192.168.0.1,192.168.0.2,192.168.0.3 sequre my_protocol.codon <pid>
```

---

## Runtime initialization

_Defined in `stdlib/sequre/runtime.codon`_

These functions set up the MPC environment for distributed (non-local) execution:

| Function | Description |
|---|---|
| `mpc()` | Parse command-line args, create an MPC instance for the current party, run MHE setup. Returns the initialized environment. |

### Typical distributed entry point

```python
from sequre.runtime import mpc as init_mpc

mpc = init_mpc()
# mpc.pid is set from sys.argv
# mpc.mhe is initialized with default_setup()
```

### Typical local entry point

```python
from sequre.runtime import local

@local
def main(mpc):
    ...  # protocol logic

main()  # forks N parties automatically --- no need to pass mpc instance
```

### Typical CLI-controlled entry point

```python
from sequre.runtime import main

@main
def my_protocol(mpc):
    ...  # protocol logic

my_protocol()  # --local → forks locally; otherwise → online
```

---

## Compiler IR passes

The Sequre compiler plugin processes `@sequre`-annotated functions through these IR passes (in order):

| Pass | Description |
|---|---|
| **MHEOptimizations** | Runs first, on the functions under `@mhe_cipher_opt` only: factors and reorders their arithmetic (see above) |
| **ExpressivenessTransformations** | Rewrites standard operators on secure types to their secure equivalents |
| **MPCOptimizations** | Optimizes polynomial evaluations and MPC-specific patterns |
<!-- | **Debugger** | Inserts debug instrumentation when `@debug` is active | -->
