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
3. **MHE optimizations** (`@mhe_mat_opt`, `@mhe_cipher_opt`, `@mhe_enc_opt`) — HE-specific optimizations

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

### `@mhe_cipher_opt`

Enables ciphertext-expression optimization. The compiler reorders arithmetic expressions to maximize cheaper ciphertext-plaintext operations over expensive ciphertext-ciphertext operations.

### `@mhe_enc_opt`

Enables encoding-mode optimization. The compiler performs a brute-force search over possible encoding layouts (row-wise, column-wise, diagonal) for each matrix multiplication and selects the optimal one.

### `@debug`

Enables debug instrumentation. The compiler inserts additional logging, assertions, and timing instrumentation into the annotated function. -->

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
| **ExpressivenessTransformations** | Rewrites standard operators on secure types to their secure equivalents |
| **MPCOptimizations** | Optimizes polynomial evaluations and MPC-specific patterns |
| **MHEOptimizations** | Optimizes HE expression ordering, encoding modes, and matrix strategies |
<!-- | **Debugger** | Inserts debug instrumentation when `@debug` is active | -->
