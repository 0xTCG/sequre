#!/usr/bin/env python3
"""Differential checks of Sequre's IR optimizations: @mhe_cipher_opt, and @mhe_enc_opt (accepted, does nothing).

Each case is one function body compiled three times in one program: with the decorator under @sequre, under @sequre
alone, and as plain code that runs on the inputs in the clear. The program runs in local mode, and every compute
party checks that the decorated function gives the undecorated result and the plaintext one, costs no more (secure
multiplications and additions, MHE-MPC switches, bootstraps, matmuls and their complexity), costs less where the pass
must fire, and keeps the function's effects (printed markers). Programs are built with SEQURE_IR_OPT_DEBUG=1, and the
cases about call sites also check that each realized decorated function was planned once. The script prints one line
per case.

  scripts/ir_opt_check.py matrix  each construct a function body can hold, one program per construct and decorator
  scripts/ir_opt_check.py types   each decorator on each operand type, alone and mixed with arrays and scalars
  scripts/ir_opt_check.py gen     seeded random bodies (--seed, --count, --per-program)
  scripts/ir_opt_check.py all     the three

Options:
  --out DIR        where programs, binaries and logs go (default: a new temporary directory)
  --case TEXT      only the cases whose name contains TEXT (a generated case: gen-<family>-<seed>-<index>)
  --family NAME    only these operand families (comma-separated; see --list)
  --decorator D    only mhe_cipher_opt or mhe_enc_opt
  --jobs N         parallel builds (default 4); runs go two at a time
  --parties N      parties in local mode, CP0 included (default 3)
  --noop           compile the decorated twins without the decorator (the cases where the pass must fire then fail)
  --release        build with -release
  --list           print the cases and exit
  --emit           write the programs and exit

Programs are built with the `sequre` launcher, which honours CODON_BIN and SEQURE_PLUGIN_PATH (a plugin directory
other than the installed one). The exit status is 0 when every case passes.
"""

import argparse
import concurrent.futures
import os
import random
import re
import shutil
import subprocess
import sys
import tempfile
import time

DECORATORS = ("mhe_cipher_opt", "mhe_enc_opt")
DEFAULT_SEED = 24

# ---------------------------------------------------------------- operand families
#
# Each family gives the four operands a, b, c, d (and the plaintext arrays pa..pd the reference runs on). Square
# matrices, so that x.T fits wherever x does. MP families split the rows among the compute parties.

FAMILIES = {
    "st": dict(doc="Sharetensors (fixed point, field)", he=False, kinds="SSSS"),
    "st-ring": dict(doc="Sharetensors (fixed point, ring)", he=False, kinds="SSSS", ring=True),
    "sti": dict(doc="Sharetensors of integers (field)", he=False, kinds="SSSS", ints=True),
    "st-mix": dict(doc="Sharetensors: a, b fixed point; c, d integers (field)", he=False, kinds="SSSS", mix=True),
    "ct": dict(doc="Ciphertensor[Ciphertext] a, b; ndarrays c, d", he=True, kinds="CCNN"),
    "ct4": dict(doc="four Ciphertensor[Ciphertext]", he=True, kinds="CCCC"),
    "ct4-T": dict(doc="Ciphertensor[Ciphertext] a, b row-wise; c, d stored transposed (enc(x.T).T)", he=True,
                  kinds="CCTT"),
    "ct4-diag": dict(doc="Ciphertensor[Ciphertext] a, b row-wise; c, d diagonal-encoded", he=True, kinds="CCDD"),
    "ctpt": dict(doc="Ciphertensor[Ciphertext] a, b; Ciphertensor[Plaintext] c, d", he=True, kinds="CCPP"),
    "mpp": dict(doc="four MPPs", he=True, kinds="MMMM", mp=True),
    "mpp-uneven": dict(doc="four MPPs split unevenly (a, c: CP1 holds most rows; b, d: the last compute party)",
                       he=True, kinds="MMMM", mp=True, uneven=True),
    "mpp-nd": dict(doc="MPPs a, b; ndarrays c, d", he=True, kinds="MMNN", mp=True),
    "mpp-enc": dict(doc="four MPPs whose partitions are encrypted", he=True, kinds="EEEE", mp=True),
    "mpa": dict(doc="four MPAs", he=True, kinds="AAAA", mp=True),
    "mpa-enc": dict(doc="four MPAs encrypted at every compute party", he=True, kinds="QQQQ", mp=True),
    "mpa-encT": dict(doc="encrypted MPAs; c, d stored transposed", he=True, kinds="QQRR", mp=True),
    "mpu-encT": dict(doc="MPUs of encrypted MPAs; c, d stored transposed", he=True, kinds="WWXX", mp=True),
    "mpu-p": dict(doc="four MPUs of partitions", he=True, kinds="UUUU", mp=True),
    "mpu-a": dict(doc="four MPUs of aggregates", he=True, kinds="VVVV", mp=True),
}

TOLERANCE = {"st": 2e-3, "st-ring": 2e-3, "sti": 1e-9, "st-mix": 2e-3}


def family_setup(fam):
    """Codon lines (inside run) that define pa..pd and a..d."""
    f = FAMILIES[fam]
    lines = []
    if any(k in f["kinds"] for k in "QRWX"):
        lines.append("    R, C = 4, 4")
    elif f.get("uneven"):
        lines.append("    n_cp = mpc.comms.number_of_parties - 1")
        lines.append("    R, C = max(4, n_cp + 1), 4")
    elif f.get("mp"):
        lines.append("    n_cp = mpc.comms.number_of_parties - 1")
        lines.append("    R, C = 2 * n_cp, 4")
        lines.append("    start, stop = (0, R) if mpc.pid == 0 else (2 * (mpc.pid - 1), 2 * mpc.pid)")
    else:
        lines.append("    R, C = 4, 4")

    def ints(x):
        return bool(f.get("ints") or (f.get("mix") and x in "cd"))

    for i, x in enumerate("abcd"):
        lines.append(f"    p{x} = _mat({i}, R, C, {'True' if ints(x) else 'False'})")
    modulus = "MPC_RING_SIZE" if f.get("ring") else "MPC_FIELD_SIZE"
    for i, (x, k) in enumerate(zip("abcd", f["kinds"])):
        if k == "S":
            data = f"[[int(v) for v in row] for row in p{x}.tolist()]" if ints(x) else f"p{x}.tolist()"
            lines.append(f"    {x} = Sharetensor.enc(mpc, {data}, 0, {modulus})")
        elif k == "C":
            lines.append(f"    {x} = Ciphertensor[Ciphertext].enc(mpc, p{x}.tolist())")
        elif k == "P":
            lines.append(f"    {x} = Ciphertensor[Plaintext].enc(mpc, p{x}.tolist())")
        elif k == "T":
            lines.append(f"    {x} = Ciphertensor[Ciphertext].enc(mpc, p{x}.T.tolist()).T")
        elif k == "D":
            lines.append(f"    {x} = Ciphertensor[Ciphertext].enc(mpc, p{x}.tolist(), encoding=ENC_DIAG)")
        elif k == "E":
            lines.append(f"    {x} = MPP[Tuple[int, int], float](mpc, Ciphertensor[Ciphertext].enc(mpc, "
                         f"p{x}[start:stop] if mpc.pid else p{x}))")
        elif k == "Q":
            lines.append(f"    {x} = _enc_mpa(mpc, p{x})")
        elif k == "R":
            lines.append(f"    {x} = _enc_mpa(mpc, p{x}.T).T")
        elif k == "W":
            lines.append(f"    {x} = _enc_mpu(mpc, p{x})")
        elif k == "X":
            lines.append(f"    {x} = _enc_mpu(mpc, p{x}.T).T")
        elif k == "N":
            lines.append(f"    {x} = p{x}.copy()")
        elif k == "M":
            if f.get("uneven"):
                lines.append(f"    start, stop = _rows(mpc, R, {i})")
            lines.append(f"    {x} = MPP(mpc, p{x}[start:stop] if mpc.pid else p{x})")
        elif k == "A":
            lines.append(f"    {x} = MPA(mpc, p{x} if mpc.pid == 1 else p{x}.zeros())")
        elif k == "U":
            lines.append(f"    {x} = MPU(mpc, p{x}[start:stop] if mpc.pid else p{x}, \"partition\")")
        elif k == "V":
            lines.append(f"    {x} = MPU(mpc, p{x} if mpc.pid == 1 else p{x}.zeros(), \"additive\")")
    return lines


def arg(fam, x):
    """The argument passed for operand x: arrays are copied, so an in-place update in one call is not seen by the next."""
    return f"{x}.copy()" if FAMILIES[fam]["kinds"]["abcd".index(x)] == "N" else x


# ---------------------------------------------------------------- cases


class Case:
    def __init__(self, name, body, families=("st",), fire=False, same=False, markers=0, extra="", ref=True,
                 returns=True, decorators=DECORATORS, twice=False, generator=False, fire_for=None, generic=False,
                 delta=None, bytes_rule=0, trace=False):
        self.name = name
        self.body = body if isinstance(body, list) else body.strip("\n").split("\n")
        self.families = families
        self.fire = fire  # the decorated function must cost less (for @mhe_cipher_opt)
        self.fire_for = fire_for  # families in which it must fire (default: all of the case's)
        self.same = same  # the decorated function must cost exactly the same
        self.markers = markers  # marker lines each call prints at each party
        self.extra = extra
        self.ref = ref  # the body also runs on the plaintext inputs
        self.returns = returns  # the body returns a value
        self.decorators = decorators
        self.twice = twice  # the decorated function is called from two call sites
        self.generator = generator
        self.generic = generic  # the decorated function is also realized for the plaintext inputs
        self.delta = delta  # the decorated function's multiplications minus the undecorated ones, when fixed
        self.bytes_rule = bytes_rule  # 1: the decorated function sends fewer bytes; 2: the same bytes
        self.trace = trace  # each realized decorated function of the case is planned exactly once (build log)


def matrix_cases():
    c = []
    add = c.append
    # H-42: literals, field reads, calls, a bare return
    add(Case("literal-float-add", "return a * c + 1.0", same=True))
    # (the product of a Sharetensor by an int is local and uncounted: these two save nothing the counters see)
    add(Case("literal-int-mul", "return a * c + a * 2"))
    add(Case("literal-left", "return 2 * a + a * c"))
    add(Case("literal-only", "return a * 2 + a * 3", fire=True))
    add(Case("field-read", "bx = Box_{n}(c, d)\nreturn a * bx.w + a * bx.v", fire=True,
             extra="class Box_{n}[T]:\n    w: T\n    v: T\n\n    def __init__(self, w: T, v: T):\n"
                   "        self.w = w\n        self.v = v\n"))
    add(Case("property-T", "return a * b.T + a * c.T", fire=True))
    add(Case("two-arg-call", "return a * pick_{n}(b, c) + a * d", same=True,
             extra="def pick_{n}(x, y):\n    return x\n"))
    add(Case("min-call", "return a * c + a * min(2, 3)", same=True))
    add(Case("bare-return", "x = a * c\nreturn", same=True, returns=False, ref=False))
    add(Case("alias-assign", "x = a\nreturn x * c + x * d", fire=True))
    # H-41: definitions under control flow, reassignment
    add(Case("if-reassign", "e = a * c\nif flag:\n    e = b * d\nreturn e + a * d"))
    add(Case("for-loop", "e = a * c\nfor _ in range(2):\n    e = e + b\nreturn e * d + a * d", fire=True))
    add(Case("while-loop", "e = a * c\nk = 0\nwhile k < 2:\n    e = e + b\n    k += 1\nreturn e * d + a * d",
             fire=True))
    add(Case("break-continue", "e = a * c\nfor i in range(4):\n    if i == 1:\n        continue\n"
                               "    if i == 3:\n        break\n    e = e + b\nreturn e * d + a * d", fire=True))
    add(Case("with-block", "with mpc.randomness.global_stream():\n    x = a * c\nreturn x + a * d"))
    add(Case("try-block", "try:\n    x = a * c\nexcept:\n    x = b * d\nreturn x + a * d"))
    add(Case("raise-branch", "if not flag:\n    raise ValueError('flag')\nreturn a * c + a * d", fire=True))
    add(Case("assert-stmt", "assert flag\nreturn a * c + a * d", fire=True))
    add(Case("pass-stmt", "pass\nreturn a * c + a * d", fire=True))
    add(Case("nested-def", "def inner(x):\n    return x\nreturn a * inner(c) + a * d", same=True))
    add(Case("lambda-call", "f = lambda x: x\nreturn a * f(c) + a * d", same=True))
    add(Case("closure-capture", "e = a * c\nh = lambda: e\nr = e + a * d\nreturn r + h()"))
    add(Case("param-reassign", "e = a * c\na = d\nreturn e + a * c", fire=True))
    add(Case("walrus", "return (t := a * c) + t * d"))
    add(Case("ternary", "return (a * c if flag else a * d) + a * b"))
    add(Case("global-scalar", "return a * K_{n} + c * K_{n}", fire=True, extra="K_{n} = 1.5\n"))
    # H-40: repeated subexpressions
    add(Case("shared-sub", "e = a * b\nreturn e * c + e * d", fire=True))
    add(Case("self-add", "e = a * b\nreturn e + e", same=True))
    add(Case("shared-twice", "e = a * b\nf = e * c\nreturn f + e * d", fire=True))
    # folding copies a definition into each read: twelve definitions read twice each must not blow up the compiler
    add(Case("fold-chain", "\n".join(["e1 = a * c"] + [f"e{i} = e{i - 1} + e{i - 1}" for i in range(2, 13)] +
                                      ["return e12 * d + e12 * b"]), fire=True))
    # more than 64 fold candidates: z is read by u as well, so it must not be folded (c24-r2-3)
    add(Case("fold-64", "\n".join(["x0 = a * c", "x1 = a * d"] + [f"f{i} = a + b" for i in range(62)] +
                                    ["z = b * c", "t = x0 + x1 + z", "u = secure_operator.identity(z)", "return t + u"]),
             fire=True))
    # the test_ir_passes forms
    add(Case("compact", "return a * c + a * d + b * c + b * d", fire=True))
    add(Case("split", "e = a * c + a * d\nf = b * c + b * d\nreturn e + f", fire=True))
    add(Case("redundant", "e = a * (c + d)\nf = b * (c + d)\nh = secure_operator.identity(e)\nreturn h + f",
             same=True))
    # M-IR-01: effects and dead assignments stay
    add(Case("print-effect", "print('marker:{case}')\nreturn a * c + a * d", fire=True, markers=1))
    add(Case("effect-in-def", "unused = noisy_{n}(a, '{case}')\nreturn a * c + a * d", fire=True, markers=1,
             extra="def noisy_{n}(x, tag: str):\n    print('marker:' + tag)\n    return x\n"))
    add(Case("dead-secure", "unused = a * c\nreturn b * d", same=True))
    # (exactly one product less: the reveal's product runs, and only the sum gets cheaper)
    add(Case("reveal-effect", "r = (a * c).reveal(mpc)\nreturn a * c + a * d", fire=True, ref=False, delta=-1))
    # M-IR-04 and recursion: once per function (trace: one "planned" line per realized function in the build log)
    add(Case("two-call-sites", "return a * c + a * d", fire=True, twice=True, trace=True))
    add(Case("generic-two-types", "return a * c + a * d", fire=True, generic=True, trace=True))
    add(Case("nested-decorated", "return {inner}(mpc, a, c) + a * d", trace=True,
             extra="@sequre\ndef inner_plain_{n}(mpc, x, y):\n    return x * y + x * y\n\n"
                   "@sequre\n@{decorator}\ndef inner_opt_{n}(mpc, x, y):\n    return x * y + x * y\n\n"
                   "def inner_ref_{n}(mpc, x, y):\n    return x * y + x * y\n"))
    add(Case("recursion", "if flag:\n    return {self}(mpc, a, b, c, d, False) + a * d\nreturn a * c + a * d",
             trace=True))
    add(Case("generator", "yield a * c + a * d", generator=True, same=True))
    # @ chains (M-IR-09, L-140): only Sharetensor chains are reordered, by their shape model, when strictly cheaper
    # than written (these square ones tie, so every chain here stays as written)
    add(Case("matmul-chain", "return a @ b.T @ c", same=True))
    add(Case("matmul-chain-paren", "return a @ (b.T @ c)", same=True))
    add(Case("matmul-chain-sum", "return (a + b) @ c @ d + a @ b @ c", same=True))
    add(Case("matmul-in-sum", "return a @ b + a * c + a * d", fire=True))
    # 4 x 1 @ 1 x 4 @ 4 x 1: 32 left to right, 8 as x @ (y @ z); 3 x 4 @ 4 x 1 @ 1 x 2: 18 as written, 32 the other way
    # (which the first product's cost alone would pick). The orders differ in the bytes they send.
    add(Case("chain-st-reorder", "x = a.T[:1].T\ny = b[:1]\nz = c.T[:1].T\nreturn x @ y @ z", bytes_rule=1,
             families=("st", "st-ring")))
    add(Case("chain-st-written", "x = a[:3]\ny = b.T[:1].T\nz = c[:1].T[:2].T\nreturn x @ y @ z", bytes_rule=2,
             families=("st", "st-ring")))
    # a qualifying chain below one that does not (an operand holding a call): found and reordered (L-140). (No chain
    # ending in a vector: Sharetensor and Ciphertensor products take matrices only, so matmul_chain's vector branch
    # is defensive.)
    add(Case("chain-st-inner", "x = a.T[:1].T\ny = b[:1]\nz = c.T[:1].T\nreturn x @ y @ z @ (d[:1] + d[:1])",
             bytes_rule=1, families=("st", "st-ring")))
    # Partitions split differently (a, c one way; b, d another): expressions of several MP tensors stay as written
    unev = ("mpp-uneven",)
    add(Case("uneven-factor", "return a * c + a * d", same=True, families=unev))
    add(Case("uneven-factor2", "return a * b + c * b + a * d", same=True, families=unev))
    add(Case("uneven-split", "e = a * b + a * d\nf = c * b + c * d\nreturn e + f", same=True, families=unev))
    # MPP chains stay as written, also when the parties hold different numbers of rows (each party's own cost of
    # a @ b differs then: c24-r1-1)
    add(Case("chain-uneven", "return a @ b @ c", same=True, families=("mpp-uneven",)))
    add(Case("chain-uneven-T", "return a @ b.T @ c", same=True, families=("mpp-uneven",)))
    add(Case("chain-uneven-paren", "return a @ (b @ c)", same=True, families=("mpp-uneven",)))
    # `@` products are never evaluated more often than written; folding may merge equal ones (c24-r1-4)
    add(Case("matmul-dup", "e = a @ b\nreturn e * c + e + e + (a @ b) * d", same=True, families=("st", "ct4")))
    # (on Ciphertensors: an expression of several stays as written)
    add(Case("matmul-merge", "e = a @ b\nf = a @ b\nreturn e * c + f * d", fire=True, families=("st", "ct4"),
             fire_for=("st",)))
    # Integer public operands: never combined where that could wrap at 64 bits (c24-r1-6)
    # (on integer shares: fixed-point values that large do not reveal)
    add(Case("int-literal-wrap", "return a * 4294967296 * 4294967296", same=True, families=("sti",)))
    add(Case("int-global-wrap", "return a * K_{n} * K_{n}", same=True, extra="K_{n} = 4294967296\n",
             families=("sti",)))
    add(Case("int-public-group", "return a * (K_{n} * K_{n}) + c * d", same=True, extra="K_{n} = 3\n"))
    add(Case("int-literal-sum", "return a * 3 + a * 4 + b * c", fire=True))
    add(Case("float-public-merge", "return a * 1.5 + a * 2.5 + b * c", fire=True))
    # Ciphertensors and MP tensors stored differently: expressions of several stay as written, results stored as
    # written (a later statement may need it: setting a slice of a transposed Ciphertensor raises; c24-r2-4, -8)
    lay = ("ct4-T", "ct4-diag")
    add(Case("lay-factor", "return a * c + a * d", families=lay, same=True))
    add(Case("lay-right", "return c * a + d * a", families=lay, same=True))
    add(Case("lay-setitem", "r = a * c + a * d\ns = a * 1.0\ns[0:2] = r[0:2]\nreturn s", families=lay, ref=False,
             same=True))
    add(Case("lay-compact", "return a * c + a * d + b * c + b * d", families=lay, same=True))
    add(Case("uneven-shape", "r = a * b + a * d\nreturn r * 1.0 if r.shape[0] == a.shape[0] else r * 0.0",
             families=("mpp-uneven",)))
    add(Case("mpT-mix", "return a * b.T + a * c", families=("mpp-enc", "mpp")))
    # MPAs and MPUs of aggregates (c24-r3-3; their products aggregate and encrypt operands in place: c24-r4-3)
    add(Case("mpa-T-mix", "return a * b.T + a * c", families=("mpa-enc",), same=True))
    add(Case("mpa-T-set", "r = a * b.T + a * c\nd[0:2] = r[0:2]\nreturn d", families=("mpa-enc",), ref=False,
             same=True))
    add(Case("mpa-factor", "return a * c + a * d", families=("mpa-enc",), same=True))
    add(Case("mpa-stored-T", "return a * c + a * d", families=("mpa-encT", "mpu-encT"), same=True))
    # an MP value times a small int is additions (no level): 3 + 4 = 7 would cost one (c24-r3-4)
    add(Case("mp-int7", "return a * 3 + a * 4 + b * c", families=("mpp-enc", "mpa-enc"), same=True))
    add(Case("mp-int5", "return a * 2 + a * 3 + b * c", families=("mpp-enc", "mpa-enc")))
    add(Case("mp-T-flag", "r = a * c.T + a * d.T\nreturn r * 1.0 if r._transposed == a._transposed else r * 0.0",
             families=("mpp",), ref=False))
    # Ciphertensors stored in different layouts (x.T flips a flag): a new addition of the two would convert one
    # through MPC; several secure tensors of these kinds stay as written
    add(Case("ct-layout-mix", "return a * b.T + a * c", same=True,
             families=("ct4", "ct", "mpp", "mpa", "mpu-p", "mpu-a")))
    add(Case("ct-layout-mix2", "return b.T * a + c * a + d * c", same=True, families=("ct4",)))
    # Fixed-point flags (a, b fixed point; c, d integers): a plan over both kinds runs only when the flags agree,
    # the expression as written otherwise (c24-r1-2)
    mix = ("st-mix",)
    add(Case("fp-mix-factor", "return a * c + a * b", same=True, families=mix))
    # (a + b) * (c + d): each addition pairs two of a kind, so the plan runs; (a + c) * (b + d) does not
    add(Case("fp-mix-split", "e = a * c + a * d\nf = b * c + b * d\nreturn e + f", fire=True, families=mix))
    add(Case("fp-mix-split-bad", "e = a * c + a * b\nf = b * c + b * b\nreturn e + f", same=True, families=mix))
    # a float added to an integer Sharetensor (Sequre adds its fixed-point encoding without a word), from a product
    # with a float ndarray too (c24-r2-1, -2)
    add(Case("fp-float-cofactor", "return a * c + a * 0.5", same=True, families=mix))
    add(Case("fp-float-fixed", "return a * b + a * 0.5", fire=True, families=mix))
    add(Case("fp-float-split", "e = a * c + a * 0.5\nf = b * c + b * 0.5\nreturn e + f", same=True, families=mix))
    add(Case("fp-float-sum", "return a * c + a * d + a * 0.5", families=mix))
    add(Case("fp-ndarray-matmul", "return a * (c @ W_{n}) + a * d", same=True, families=mix,
             extra="W_{n} = array([[0.5 + 0.25 * float((i + j) % 3) for j in range(4)] for i in range(4)])\n"))
    # an int-led public sum with a float, (2 * 3) + 0.5, is a float too; a float meeting an integer Sharetensor
    # through literals; a definition read twice folded into a plan whose check fails runs once (c24-r3-1, -7)
    add(Case("fp-intlead", "return a * 2 * 3 + a * 0.5 + a * c", same=True, families=mix))
    add(Case("fp-float-intlit", "return c + 2 + 0.5", same=True, ref=False, families=mix))
    add(Case("fp-fold-twice", "e = a * c\nreturn e * b + e * d", same=True, families=mix))
    # combined float constants next to Sharetensors stay within the encoding below as well (c24-r3-6)
    add(Case("fp-underflow", "return a * 0.00001 * 0.00001 + b * c", families=("st",)))
    # constants next to Sharetensors stay within the 31 integer bits of Sequre's encoding of a public value (c24-r2-6)
    add(Case("fp-const-range", "x = a * 0.0 - 2000000000.0\nreturn x + 2000000000 + 2000000000", same=True,
             families=("st",)))
    add(Case("fp-mix-ints", "return c * d + c * c", fire=True, families=mix))
    add(Case("fp-mix-fixed", "return a * b + a * a", fire=True, families=mix))
    # H-43: statements @mhe_enc_opt meets (in "ct", a is a Ciphertensor and c, d are ndarrays: its candidates)
    enc = ("st", "ct")
    add(Case("enc-assign", "x = a @ c.T\nreturn x", families=enc))
    add(Case("enc-alias", "x = a\ny = c\nreturn x @ y.T", families=enc))
    add(Case("enc-if", "x = a @ c\nif flag:\n    x = a @ d\nreturn x", families=enc))
    add(Case("enc-for", "x = a @ c\nfor _ in range(2):\n    x = x + a @ d\nreturn x", families=enc))
    add(Case("enc-with", "with mpc.randomness.global_stream():\n    x = a @ c.T\nreturn x", families=enc))
    add(Case("enc-bare-return", "x = a @ c.T\nreturn", returns=False, ref=False, same=True, families=enc))
    add(Case("enc-print", "print('marker:{case}')\nreturn a @ c.T + (c @ a.T).T", markers=1, families=enc))
    add(Case("enc-chain", "return a @ c @ d.T", families=enc))
    add(Case("enc-left", "return c @ a + d @ a", families=enc))
    add(Case("enc-reassign-param", "x = a @ c\nc = d\nreturn x + a @ c", families=enc))
    return c


def types_cases():
    # Each decorator on each operand family. Cases whose operator combinations a family lacks are left out of it.
    # (A Ciphertensor[Plaintext] works with Ciphertensor[Ciphertext] operands only: Sequre raises on any other, so
    # ctpt cases pair c and d with a and b only, and the planner leaves them as written)
    # Expressions of two or more secure Ciphertensors or MP tensors stay as written (their storage is not in their
    # types, and their products change operands in place): the pass fires on Sharetensors, and on one secure tensor of
    # the other kinds with public operands (the mixed families' a with c and d)
    shares = ("st", "st-ring", "sti")
    secure = shares + ("ct4", "mpp", "mpa", "mpu-p", "mpu-a")
    mixed = ("ct", "mpp-nd")
    allf = secure + mixed + ("ctpt",)
    c = []
    add = c.append
    add(Case("t-factor", "return a * c + a * d", families=allf, fire=True, fire_for=shares + mixed))
    add(Case("t-compact", "return a * c + a * d + b * c + b * d", families=allf, fire=True, fire_for=shares))
    add(Case("t-reorder", "return a * b + a * c + a * d", families=allf, fire=True, fire_for=shares))
    add(Case("t-public-sum", "return a + c + b + d", families=allf))
    add(Case("t-public-prod", "return a * c * b * d", families=allf))
    # Products of a Sharetensor or an MPA by an int are local and uncounted, so the counters see no saving there
    # (factoring 2 out of a * 2 + c * 2 saves a counted product only where c is secure too)
    add(Case("t-scalar-right", "return a * 2 + c * 2", families=tuple(f for f in allf if f != "ctpt")))
    # An int on the left of a Ciphertensor (M-IR-07) or an MPA does not compile under @sequre alone
    add(Case("t-scalar-left", "return 2 * a + 2 * c", families=tuple(f for f in allf if f not in
                                                                         ("ct", "ct4", "ctpt", "mpa"))))
    # An ndarray on the left of an MPP does not compile under @sequre alone (multiparty_partition.codon's
    # ndarray.__mul__)
    # (a Ciphertensor[Plaintext] on the left of a product is typed as a plaintext, and the sum of two such products
    # raises under @sequre alone: ctpt keeps its plaintexts on the right)
    add(Case("t-array-left", "return c * a + d * a", families=("ct",), fire=True))
    add(Case("t-public-factor", "return c * a + c * b", families=("ct4", "ct"), same=True))
    add(Case("t-right-factor", "return a * c + b * c", families=("ctpt", "ct4", "ct"), same=True))
    # (in the mixed families e and f are planned on their own: a * (c + d), b * (c + d))
    add(Case("t-split", "e = a * c + a * d\nf = b * c + b * d\nreturn e + f", families=allf, fire=True,
             fire_for=shares + mixed))
    add(Case("t-matmul", "return a @ b.T + (b @ a.T).T", families=("ct", "ct4", "mpp", "mpu-p", "st")))
    return c


# ---------------------------------------------------------------- the seeded generator

GEN_FAMILIES = ("st", "st-ring", "sti", "ct", "ct4", "mpp", "mpu-p")


class Gen:
    """Random bodies over a, b, c, d: sums and products with repeated subexpressions, literals, intermediate
    assignments, reassignments, branches, loops, in-place updates of arrays and calls that stop folding. Operand
    orders Sequre itself rejects under @sequre alone are avoided: a literal on the left of a secure Ciphertensor, MPP
    or MPA (M-IR-07), an array on the left of an MPP, and subtraction among integers (negative results reveal as field
    elements)."""

    NO_SCALAR_LEFT = ("ct", "ct4", "ctpt", "mpp", "mpp-nd", "mpa")
    NO_ARRAY_LEFT = ("mpp-nd",)

    def __init__(self, rng, fam):
        self.rng = rng
        self.fam = fam
        kinds = FAMILIES[fam]["kinds"]
        self.secret = [x for x, k in zip("abcd", kinds) if k in "SCMAUV"]
        self.public = [x for x, k in zip("abcd", kinds) if k in "NP"]
        self.arrays = [x for x, k in zip("abcd", kinds) if k == "N"]
        self.ints = FAMILIES[fam].get("ints", False)
        self.temps = []

    def literal(self):
        if self.ints:
            return self.rng.choice(["2", "3"])
        return self.rng.choice(["2", "3", "0.5", "1.5"])

    def leaf(self, need_secret=False):
        # (text, kind): kind S secure, N public tensor, L scalar
        r = self.rng.random()
        if need_secret or r < 0.55:
            return self.rng.choice(self.secret + self.temps), "S"
        if r < 0.8 and self.public:
            return self.rng.choice(self.public), "N"
        return self.literal(), "L"

    def join(self, l, op, r):
        # l op r, with the operands swapped where Sequre rejects the order (op commutes)
        if r[1] == "S" and ((l[1] == "L" and self.fam in self.NO_SCALAR_LEFT) or
                            (l[1] == "N" and self.fam in self.NO_ARRAY_LEFT)):
            l, r = r, l
        kind = "S" if "S" in (l[1], r[1]) else ("N" if "N" in (l[1], r[1]) else "L")
        return f"{l[0]} {op} {r[0]}", kind

    def expr(self, depth, need_secret=True):
        r = self.rng.random()
        if depth <= 0 or r < 0.2:
            return self.leaf(need_secret)
        if r < 0.45:
            # a factorable sum: x * y1 + x * y2 (+ x * y3)
            x = self.expr(depth - 1, need_secret)
            terms = []
            for _ in range(self.rng.choice([2, 2, 3])):
                y = self.expr(depth - 2, False)
                terms.append(self.join(x, "*", y) if self.rng.random() < 0.7 else self.join(y, "*", x))
            # (a + b + c is (a + b) + c: secure terms first, so no scalar or array comes left of a secure term)
            terms.sort(key=lambda t: "SNL".index(t[1]))
            kind = terms[0][1]
            return "(" + " + ".join(t for t, _ in terms) + ")", kind
        op = self.rng.choice(["+", "*", "*", "+"] + ([] if self.ints else ["-"]))
        left_secret = need_secret and (op == "-" or self.rng.random() < 0.7)
        if op == "-" and not need_secret:
            op = "+"
        l = self.expr(depth - 1, left_secret)
        rr = self.expr(depth - 1, need_secret and not left_secret)
        if op == "-":
            return f"({l[0]} - {rr[0]})", "S"
        t, kind = self.join(l, op, rr)
        return f"({t})", kind

    def body(self):
        lines = []
        n = self.rng.randint(0, 4)
        for i in range(n):
            kind = self.rng.random()
            if kind < 0.45 or not self.temps:
                t = f"t{len(self.temps)}"
                lines.append(f"{t} = {self.expr(2)[0]}")
                self.temps.append(t)
            elif kind < 0.6:
                t = self.rng.choice(self.temps)
                lines.append(f"{t} = {self.expr(2)[0]}")
            elif kind < 0.7:
                t = self.rng.choice(self.temps)
                lines += ["if flag:", f"    {t} = {self.expr(1)[0]}"]
            elif kind < 0.8:
                t = self.rng.choice(self.temps)
                e = self.join((t, "S"), "+", self.expr(1))[0]
                lines += ["for _ in range(2):", f"    {t} = {e}"]
            elif kind < 0.9 and self.arrays:
                p = self.rng.choice(self.arrays)
                lines.append(f"{p} += {self.rng.choice(['0.5', '1.5'])}")
            else:
                t = self.rng.choice(self.temps)
                lines.append(f"{t} = secure_operator.identity({t})")
        lines.append(f"return {self.expr(3)[0]}")
        return lines


def gen_cases(seed, count, families):
    cases = []
    for fam in families:
        rng = random.Random(f"{seed}-{fam}")
        for i in range(count):
            body = Gen(rng, fam).body()
            cases.append(Case(f"gen-{fam}-{seed}-{i}", body, families=(fam,), decorators=("mhe_cipher_opt",)))
    return cases


# ---------------------------------------------------------------- programs

HEADER = '''# Generated by scripts/ir_opt_check.py; do not edit.
import sys

import secure_operator

from sequre import main, sequre, mhe_cipher_opt, mhe_enc_opt
from sequre.constants import MPC_FIELD_SIZE, MPC_RING_SIZE, ENC_DIAG
from sequre.lattiseq.ckks import Ciphertext, Plaintext
from sequre.stdlib.numpy.create import array
from sequre.stdlib.numpy.ndarray import ndarray
from sequre.types.sharetensor import Sharetensor
from sequre.types.ciphertensor import Ciphertensor
from sequre.types.multiparty_partition import MPP
from sequre.types.multiparty_aggregate import MPA
from sequre.types.multiparty_union import MPU


def _mat(k: int, rows: int, cols: int, ints: bool):
    # Values every party knows: in [0.25, 1.5], or small integers
    out = []
    for i in range(rows):
        row = []
        for j in range(cols):
            v = (3 * i + 5 * j + 7 * k) % 11
            row.append(float(1 + v % 5) if ints else 0.25 + v / 8.0)
        out.append(row)
    return array(out)


def _enc_mpa(mpc, p: ndarray[Tuple[int, int], float]) -> MPA[Tuple[int, int], float]:
    # Each compute party's share encrypted (CP1 holds the value, the others zeros); CP0 a placeholder in the clear
    if mpc.pid == 0:
        return MPA[Tuple[int, int], float](mpc, p.zeros())
    share = p if mpc.pid == 1 else p.zeros()
    return MPA[Tuple[int, int], float](mpc, Ciphertensor[Ciphertext].enc(mpc, share))


def _enc_mpu(mpc, p: ndarray[Tuple[int, int], float]) -> MPU[Tuple[int, int], float]:
    return MPU[Tuple[int, int], float](_mpp=MPP[Tuple[int, int], float](mpc), _mpa=_enc_mpa(mpc, p))


def _stats(mpc):
    s = mpc.stats
    return (s.secure_mul_count, s.secure_add_count, s.secure_mhe_mpc_switch_count, s.secure_bootstrap_count,
            s.secure_matmul_count, s.secure_matmul_complexity, s.bytes_sent)


def _rows(mpc, R: int, i: int):
    # The party's rows of operand i in an uneven family: one compute party (CP1 for a and c, the last for b and d)
    # holds all rows but one per other compute party
    n_cp = mpc.comms.number_of_parties - 1
    if mpc.pid == 0:
        return 0, R
    big = R - (n_cp - 1)
    owner = 1 if i % 2 == 0 else n_cp
    start = 0
    for p in range(1, mpc.pid):
        start += big if p == owner else 1
    return start, start + (big if mpc.pid == owner else 1)


def _reveal(mpc, x):
    if isinstance(x, ndarray) or isinstance(x, float) or isinstance(x, int):
        return x
    else:
        return x.reveal(mpc)


def _flat(x) -> List[float]:
    out = List[float]()
    if isinstance(x, ndarray):
        for v in x.flatten().tolist():
            out.append(float(v))
    else:
        out.append(float(x))
    return out


def _err(x: List[float], y: List[float]) -> float:
    # The largest difference relative to max(1, |y|), or 1e30 for different sizes
    if len(x) != len(y):
        return 1e30
    worst = 0.0
    for i in range(len(x)):
        d = abs(x[i] - y[i]) / max(1.0, abs(y[i]))
        if d != d:
            return 1e30
        if d > worst:
            worst = d
    return worst


def _delta(s0, s1):
    return (s1[0] - s0[0], s1[1] - s0[1], s1[2] - s0[2], s1[3] - s0[3], s1[4] - s0[4], s1[5] - s0[5], s1[6] - s0[6])


def _check(mpc, name: str, rp, ro, rr, dp, do, fire: bool, same: bool, tol: float, delta: int, bytes_rule: int):
    # Reveals are collective: every party takes part, CP0 (placeholders) checks nothing
    vp = _flat(_reveal(mpc, rp))
    vo = _flat(_reveal(mpc, ro))
    if mpc.pid == 0:
        return
    vr = _flat(rr)
    _verdict(mpc, name, _err(vo, vp), _err(vo, vr), _err(vp, vr), dp, do, fire, same, tol, delta, bytes_rule)


def _check_noref(mpc, name: str, rp, ro, dp, do, fire: bool, same: bool, tol: float, delta: int, bytes_rule: int):
    vp = _flat(_reveal(mpc, rp))
    vo = _flat(_reveal(mpc, ro))
    if mpc.pid == 0:
        return
    _verdict(mpc, name, _err(vo, vp), 0.0, 0.0, dp, do, fire, same, tol, delta, bytes_rule)


def _verdict(mpc, name: str, e_po: float, e_r: float, e_pr: float, dp, do, fire: bool, same: bool, tol: float,
             delta: int, bytes_rule: int):
    # dp, do: the undecorated and the decorated function's (multiplications, additions, MHE-MPC switches, bootstraps,
    # matmuls, matmul complexity, bytes sent). delta: the decorated multiplications minus the undecorated ones, or 99
    # for any. bytes_rule: 1 when the decorated function must send fewer bytes, 2 the same, 0 either.
    problems = List[str]()
    if e_po > tol:
        problems.append("decorated-vs-undecorated")
    if e_r > tol:
        problems.append("decorated-vs-plaintext")
    if e_pr > tol:
        problems.append("undecorated-vs-plaintext")
    if do[0] > dp[0] or do[1] > dp[1]:
        problems.append("costs-more")
    if do[2] > dp[2]:
        problems.append("more-switches")
    if do[3] > dp[3]:
        problems.append("more-bootstraps")
    if do[4] > dp[4]:
        problems.append("more-matmuls")
    if do[5] > dp[5]:
        problems.append("more-matmul-complexity")
    if same and (do[0] != dp[0] or do[1] != dp[1] or do[4] != dp[4] or do[5] != dp[5]):
        problems.append("cost-changed")
    if fire and not (do[4] < dp[4] or (do[4] == dp[4] and (do[0] < dp[0] or (do[0] == dp[0] and do[1] < dp[1])))):
        problems.append("did-not-fire")
    if delta != 99 and do[0] - dp[0] != delta:
        problems.append("wrong-delta")
    if bytes_rule == 1 and not do[6] < dp[6]:
        problems.append("not-fewer-bytes")
    if bytes_rule == 2 and do[6] != dp[6]:
        problems.append("bytes-changed")
    status = "ok" if len(problems) == 0 else "FAIL"
    line = ("CP" + str(mpc.pid) + ": " + status + " " + name + " err=" + str(e_po) + "/" + str(e_r) +
            " muls=" + str(dp[0]) + "->" + str(do[0]) + " adds=" + str(dp[1]) + "->" + str(do[1]) +
            " switches=" + str(dp[2]) + "->" + str(do[2]) + " bootstraps=" + str(dp[3]) + "->" + str(do[3]) +
            " matmuls=" + str(dp[4]) + "->" + str(do[4]) + " complexity=" + str(dp[5]) + "->" + str(do[5]) +
            (" bytes=" + str(dp[6]) + "->" + str(do[6]) if bytes_rule else "") +
            (" " + ",".join(problems) if len(problems) else ""))
    print(line)
    sys.stdout.flush()


def _check_none(mpc, name: str, dp, do, same: bool):
    if mpc.pid == 0:
        return
    bad = (do[0] > dp[0] or do[1] > dp[1] or do[4] > dp[4] or
           (same and (do[0] != dp[0] or do[1] != dp[1] or do[4] != dp[4])))
    line = ("CP" + str(mpc.pid) + ": " + ("FAIL" if bad else "ok") + " " + name + " muls=" + str(dp[0]) + "->" +
            str(do[0]) + " adds=" + str(dp[1]) + "->" + str(do[1]) + " matmuls=" + str(dp[4]) + "->" + str(do[4]))
    print(line)
    sys.stdout.flush()
'''


def indent(lines, n=4):
    return [(" " * n + l) if l else l for l in lines]


def subst(text, case, idx, decorator, role):
    names = {"plain": f"plain_{idx}", "opt": f"opt_{idx}", "ref": f"ref_{idx}"}
    inner = {"plain": f"inner_plain_{idx}", "opt": f"inner_opt_{idx}", "ref": f"inner_ref_{idx}"}
    return (text.replace("{n}", str(idx)).replace("{case}", case.name).replace("{decorator}", decorator)
            .replace("{self}", names[role]).replace("{inner}", inner[role]))


NOOP = False  # --noop: the decorated twin without its decorator (a pass that does nothing must fail the checks)


def program(entries, decorator, fam):
    """entries: list of (idx, case)"""
    out = [HEADER]
    for idx, case in entries:
        if case.extra:
            out.append(subst(case.extra, case, idx, decorator, "plain"))
        for role in ("ref", "plain", "opt"):
            body = [subst(l, case, idx, decorator, role) for l in case.body]
            head = f"def {'ref' if role == 'ref' else role}_{idx}(mpc, a, b, c, d, flag: bool):"
            decos = []
            if role != "ref":
                decos.append("@sequre")
            if role == "opt" and not NOOP:
                decos.append(f"@{decorator}")
            out.append("\n".join(decos + [head] + indent(body)) + "\n")
    run = ["@main", "def run(mpc):"] + family_setup(fam)
    if FAMILIES[fam].get("ring"):
        run.append("    with mpc.default_modulus(MPC_RING_SIZE):")
        pre = "        "
    else:
        pre = "    "
    tol = TOLERANCE.get(fam, 2e-3)
    args = ", ".join(arg(fam, x) for x in "abcd")
    for idx, case in entries:
        fire = case.fire and decorator == "mhe_cipher_opt" and (case.fire_for is None or fam in case.fire_for)
        # (@mhe_enc_opt is accepted and does nothing: its twin must cost exactly the same)
        same = case.same or decorator == "mhe_enc_opt"
        call = (lambda f: f"[v for v in {f}(mpc, {args}, True)][0]") if case.generator else \
            (lambda f: f"{f}(mpc, {args}, True)")
        run.append(f"{pre}s0 = _stats(mpc)")
        run.append(f"{pre}rp_{idx} = {call(f'plain_{idx}')}")
        run.append(f"{pre}s1 = _stats(mpc)")
        run.append(f"{pre}ro_{idx} = {call(f'opt_{idx}')}")
        run.append(f"{pre}s2 = _stats(mpc)")
        if case.ref:
            refargs = ", ".join(f"p{x}.copy()" for x in "abcd")
            if case.generator:
                run.append(f"{pre}rr_{idx} = [v for v in ref_{idx}(mpc, {refargs}, True)][0]")
            else:
                run.append(f"{pre}rr_{idx} = ref_{idx}(mpc, {refargs}, True)")
        delta = 99 if case.delta is None or decorator != "mhe_cipher_opt" else case.delta
        bytes_rule = case.bytes_rule if decorator == "mhe_cipher_opt" else (2 if case.bytes_rule else 0)

        def check(label, ro, d_o):
            if not case.returns:
                return f"{pre}_check_none(mpc, '{label}', _delta(s0, s1), {d_o}, {same})"
            if case.ref:
                return (f"{pre}_check(mpc, '{label}', rp_{idx}, {ro}, rr_{idx}, _delta(s0, s1), {d_o}, {fire}, "
                        f"{same}, {tol}, {delta}, {bytes_rule})")
            return (f"{pre}_check_noref(mpc, '{label}', rp_{idx}, {ro}, _delta(s0, s1), {d_o}, {fire}, {same}, "
                    f"{tol}, {delta}, {bytes_rule})")
        run.append(check(case.name, f"ro_{idx}", "_delta(s1, s2)"))
        if case.generic:
            # the same decorated function realized for ndarrays (nothing secure: left as written, plain numpy)
            refargs = ", ".join(f"p{x}.copy()" for x in "abcd")
            run.append(f"{pre}rg_{idx} = opt_{idx}(mpc, {refargs}, True)")
            run.append(f"{pre}if mpc.pid:")
            run.append(f"{pre}    _verdict(mpc, '{case.name}:plaintext-realization', 0.0, _err(_flat(rg_{idx}), "
                       f"_flat(rr_{idx})), 0.0, (0, 0, 0, 0, 0, 0, 0), (0, 0, 0, 0, 0, 0, 0), False, False, {tol}, "
                       f"99, 0)")
        if case.twice:
            run.append(f"{pre}t0 = _stats(mpc)")
            run.append(f"{pre}rt_{idx} = {call(f'opt_{idx}')}")
            run.append(f"{pre}t1 = _stats(mpc)")
            run.append(check(case.name + ":second-site", f"rt_{idx}", "_delta(t0, t1)"))
    out.append("\n".join(run) + "\n")
    out.append("\nif __name__ == \"__main__\":\n    run()\n")
    return "\n\n".join(out)


# ---------------------------------------------------------------- build and run


class Job:
    def __init__(self, key, fam, decorator, entries, out_dir):
        self.key = key
        self.fam = fam
        self.decorator = decorator
        self.entries = entries
        self.dir = os.path.join(out_dir, key)
        self.src = os.path.join(self.dir, key + ".codon")
        self.bin = os.path.join(self.dir, key)
        self.build_rc = None
        self.build_s = 0.0
        self.run_rc = None
        self.output = ""


def build(job, release):
    os.makedirs(job.dir, exist_ok=True)
    with open(job.src, "w") as f:
        f.write(program(job.entries, job.decorator, job.fam))
    cmd = ["sequre", "build"] + (["-release"] if release else []) + [job.src, "-o", job.bin]
    env = dict(os.environ, SEQURE_IR_OPT_DEBUG="1")
    t = time.time()
    with open(os.path.join(job.dir, "build.log"), "w") as log:
        try:
            job.build_rc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, env=env,
                                          timeout=1800).returncode
        except subprocess.TimeoutExpired:
            job.build_rc = "timeout"
    job.build_s = time.time() - t
    if job.build_rc == 0 and not os.path.exists(job.bin):
        job.build_rc = "no-binary"
    return job


def run(job, parties):
    rd = os.path.join(job.dir, "run")
    shutil.rmtree(rd, ignore_errors=True)
    os.makedirs(rd)
    flags = ["--local"] + ([] if FAMILIES[job.fam]["he"] else ["--skip-mhe-setup"])
    env = dict(os.environ, SEQURE_CP_COUNT=str(parties))
    with open(os.path.join(job.dir, "run.log"), "w") as log:
        try:
            job.run_rc = subprocess.run([job.bin] + flags, cwd=rd, env=env, stdout=log, stderr=subprocess.STDOUT,
                                        timeout=1800).returncode
        except subprocess.TimeoutExpired:
            job.run_rc = "timeout"
    with open(os.path.join(job.dir, "run.log"), errors="replace") as log:
        job.output = log.read()
    return job


def verdicts(job, parties):
    """One (case name, status, detail) per case of the job."""
    res = []
    if job.build_rc != 0:
        log = open(os.path.join(job.dir, "build.log"), errors="replace").read().strip().splitlines()
        tail = " | ".join(l.strip() for l in log[-3:])[:300]
        for _, case in job.entries:
            res.append((case.name, "BUILD-FAIL", f"rc={job.build_rc} {tail}"))
        return res
    lines = job.output.splitlines()
    for idx, case in job.entries:
        names = ([case.name] + ([case.name + ":second-site"] if case.twice else []) +
                 ([case.name + ":plaintext-realization"] if case.generic else []))
        problems = []
        details = []
        for nm in names:
            pat = re.compile(r"^CP(\d+): (ok|FAIL) " + re.escape(nm) + r"( |$)")
            seen = {}
            for l in lines:
                m = pat.match(l)
                if m:
                    seen.setdefault(int(m.group(1)), []).append((m.group(2), l))
            for p in range(1, parties):
                if p not in seen:
                    problems.append(f"{nm}: no line from CP{p}")
                elif any(s != "ok" for s, _ in seen[p]):
                    problems.append(next(l for s, l in seen[p] if s != "ok"))
            if 1 in seen:
                details.append(seen[1][0][1].split(" ", 3)[-1])
        if case.trace and job.decorator == "mhe_cipher_opt" and not NOOP:
            # each realized decorated function of the case planned exactly once
            log = open(os.path.join(job.dir, "build.log"), errors="replace").read()
            planned = {}
            for m in re.finditer(r"^\[mhe_cipher_opt\] function (\S+) \((\d+)\): planned$", log, re.M):
                if m.group(1) in (f"opt_{idx}", f"inner_opt_{idx}"):
                    planned[m.group(2)] = planned.get(m.group(2), 0) + 1
            if not planned:
                problems.append("no trace of the decorated function being planned")
            elif any(n != 1 for n in planned.values()):
                problems.append("planned more than once: " + ", ".join(f"{k}x{n}" for k, n in planned.items()))
        if case.markers:
            calls = 2 + (1 if case.ref else 0) + (1 if case.twice else 0)
            want = case.markers * calls * parties
            got = sum(1 for l in lines if l.strip() == f"marker:{case.name}")
            if got != want:
                problems.append(f"markers {got} (want {want})")
        if job.run_rc != 0:
            problems.insert(0, f"run rc={job.run_rc}")
        status = "ok" if not problems else "FAIL"
        res.append((case.name, status, "; ".join(problems) if problems else " ".join(details)[:300]))
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("what", choices=["matrix", "types", "gen", "all"])
    ap.add_argument("--out")
    ap.add_argument("--case")
    ap.add_argument("--family")
    ap.add_argument("--decorator", choices=DECORATORS)
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--parties", type=int, default=3)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--count", type=int, default=40, help="generated cases per family")
    ap.add_argument("--per-program", type=int, default=20, help="generated cases per program")
    ap.add_argument("--gen-families", default=",".join(GEN_FAMILIES))
    ap.add_argument("--split", action="store_true", help="one case per program (a crash of the compiler then "
                                                           "fails that case alone)")
    ap.add_argument("--noop", action="store_true", help="compile the decorated twins without the decorator: the "
                                                         "cases where the pass must fire then fail")
    ap.add_argument("--release", action="store_true")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--emit", action="store_true")
    a = ap.parse_args()
    global NOOP
    NOOP = a.noop

    jobs = []
    out = a.out or tempfile.mkdtemp(prefix="ir_opt_check.")
    os.makedirs(out, exist_ok=True)
    fams = set(a.family.split(",")) if a.family else None
    decs = (a.decorator,) if a.decorator else DECORATORS

    def keep(case, fam):
        return (not a.case or a.case in case.name) and (fams is None or fam in fams)

    if a.what in ("matrix", "all"):
        for i, case in enumerate(matrix_cases()):
            for fam in case.families:
                for d in case.decorators:
                    if d in decs and keep(case, fam):
                        jobs.append(Job(f"m-{case.name}-{fam}-{d}", fam, d, [(i, case)], out))
    if a.what in ("types", "all"):
        tc = types_cases()
        for fam in FAMILIES:
            for d in decs:
                entries = [(i, c) for i, c in enumerate(tc) if fam in c.families and keep(c, fam)]
                if a.split:
                    for i, c in entries:
                        jobs.append(Job(f"t-{c.name}-{fam}-{d}", fam, d, [(i, c)], out))
                elif entries:
                    jobs.append(Job(f"t-{fam}-{d}", fam, d, entries, out))
    if a.what in ("gen", "all"):
        gc = gen_cases(a.seed, a.count, a.gen_families.split(","))
        by_fam = {}
        for i, c in enumerate(gc):
            if keep(c, c.families[0]) and "mhe_cipher_opt" in decs:
                by_fam.setdefault(c.families[0], []).append((i, c))
        per = 1 if a.split else a.per_program
        for fam, entries in by_fam.items():
            for k in range(0, len(entries), per):
                jobs.append(Job(f"g-{fam}-{a.seed}-{k // per}", fam, "mhe_cipher_opt", entries[k:k + per], out))

    if a.list:
        for j in jobs:
            for _, c in j.entries:
                print(f"{j.key}\t{c.name}\t" + " / ".join(c.body))
        return 0
    if a.emit:
        for j in jobs:
            os.makedirs(j.dir, exist_ok=True)
            with open(j.src, "w") as f:
                f.write(program(j.entries, j.decorator, j.fam))
        print(f"{len(jobs)} programs in {out}")
        return 0

    print(f"# {len(jobs)} programs in {out}; plugin: {os.environ.get('SEQURE_PLUGIN_PATH', 'installed')}", flush=True)
    failed = 0
    total = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, a.jobs)) as bpool, \
            concurrent.futures.ThreadPoolExecutor(max_workers=2) as rpool:
        built = [bpool.submit(build, j, a.release) for j in jobs]
        runs = []
        for fut in concurrent.futures.as_completed(built):
            j = fut.result()
            if j.build_rc == 0:
                runs.append(rpool.submit(run, j, a.parties))
            else:
                for name, status, detail in verdicts(j, a.parties):
                    total += 1
                    failed += 1
                    print(f"{name}\t{j.fam}\t{j.decorator}\t{status}\t{detail}", flush=True)
        for fut in concurrent.futures.as_completed(runs):
            j = fut.result()
            for name, status, detail in verdicts(j, a.parties):
                total += 1
                failed += status != "ok"
                print(f"{name}\t{j.fam}\t{j.decorator}\t{status}\t{detail}", flush=True)
    print(f"# {total - failed} of {total} cases pass ({out})", flush=True)
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
