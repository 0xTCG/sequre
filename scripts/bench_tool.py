#!/usr/bin/env python3
"""
Helpers behind scripts/bench.sh: the case registry, the run plan, the build cache key, the completeness check of
the per-party result files, the manifest and the per-case summary. Standard library only.

    bench_tool.py list [--cases P1,P2] [--skip P1,P2]           cases (binary, flags, plan, description)
    bench_tool.py plan --cases ... --skip ... --binaries mpc,mhe,btp
                                                                  lines "<binary> main <patterns>" / "<binary> iso <name>"
    bench_tool.py cachekey --binary mpc                          build cache key (sources + stdlib + compiler)
    bench_tool.py check --dir SEG --binary mpc --cases a$,b$     completeness of SEG/CP*.tsv (JSON on stdout)
    bench_tool.py event --dir RES key=value ...                  append an event to RES/events.jsonl
    bench_tool.py manifest --dir RES --label L [--argv ...]      write RES/manifest.json
    bench_tool.py summary --dir RES [--party max|1|2]             write RES/summary.tsv and print it

Case names select by prefix; a trailing '$' asks for an exact name. Registry flags (tests/bench/bench_*.codon):
"" normal, "iso" run in its own process after the rest, "skip:<reason>" known broken (only when named exactly,
and then in its own process), "always" runs whenever its binary runs (MHE key generation, bootstrapping setup).
"""
import argparse
import glob
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BENCH_DIR = os.path.join(REPO, "tests", "bench")
BINARIES = ["mpc", "mhe", "btp"]  # build and run order
CASE_RE = re.compile(r'^\s*\(\s*"([^"]+)"\s*,\s*"([^"]*)"\s*,\s*"([^"]*)"\s*\)\s*,?\s*(#.*)?$')
# Cases that run whenever their binary runs (their work is needed by the other cases).
ALWAYS = {"mhe.setup.keygen", "btp.setup.keys", "btp.setup.bootstrapper"}
NUM_COLS = ["reps", "median_s", "min_s", "mean_s", "max_s", "q1_s", "q3_s", "mad_rel", "warm_s",
            "bytes_per_rep", "check", "wall_s"]


# ------------------------------------------------------------------------------------------------------------------
# Registry and plan
# ------------------------------------------------------------------------------------------------------------------

def source_of(binary):
    return os.path.join(BENCH_DIR, f"bench_{binary}.codon")


def registry():
    """[(binary, name, flags, description)] in registry order."""
    out = []
    for b in BINARIES:
        inside = False
        with open(source_of(b)) as f:
            for line in f:
                if line.startswith("CASES = ["):
                    inside = True
                    continue
                if inside and line.startswith("]"):
                    break
                if inside:
                    m = CASE_RE.match(line)
                    if m:
                        flags = m.group(2)
                        if m.group(1) in ALWAYS:
                            flags = flags or "always"
                        out.append((b, m.group(1), flags, m.group(3)))
    return out


def split_csv(s):
    return [p.strip() for p in (s or "").split(",") if p.strip()]


def matches(name, pattern):
    if pattern.endswith("$"):
        return name == pattern[:-1]
    return name.startswith(pattern)


def plan(cases_arg, skip_arg, binaries):
    """{binary: {"main": [names], "iso": [names], "skipped": [(name, flags)]}}; raises on unknown patterns."""
    inc, exc = split_csv(cases_arg), split_csv(skip_arg)
    reg = [r for r in registry() if r[0] in binaries]
    for p in inc:
        if not any(matches(name, p) for _, name, _, _ in reg):
            raise SystemExit(f"bench: --cases pattern '{p}' matches no case (see: scripts/bench.sh list)")
    out = {b: {"main": [], "iso": [], "skipped": []} for b in binaries}
    for b, name, flags, _ in reg:
        hit = (not inc or any(matches(name, p) for p in inc)) and not any(matches(name, p) for p in exc)
        if flags.startswith("skip"):
            if any(p == name + "$" for p in inc):
                out[b]["iso"].append(name)
            elif hit:
                out[b]["skipped"].append((name, flags))
            continue
        if not hit:
            continue
        if flags == "iso":
            out[b]["iso"].append(name)
        else:
            out[b]["main"].append(name)
    # "always" cases (key generation, bootstrapping setup) run in every process of their binary whatever the
    # selection; check_segment() expects them, so they need not be listed here.
    return out


def always_cases(binary):
    return [name for b, name, flags, _ in registry() if b == binary and flags == "always"]


# ------------------------------------------------------------------------------------------------------------------
# Result files
# ------------------------------------------------------------------------------------------------------------------

def read_tsv(path):
    """(header_meta: dict, rows: [dict], done: bool, notes: [str])."""
    meta, rows, done, notes = {}, [], False, []
    if not os.path.exists(path):
        return meta, rows, done, notes
    with open(path) as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            if line.startswith("# sequre-bench"):
                for part in line[2:].split("|"):
                    if "=" in part:
                        k, v = part.split("=", 1)
                        meta[k.strip()] = v.strip()
                continue
            if line.startswith("#"):
                notes.append(line)
                continue
            if line.startswith("case\t"):
                continue
            cols = line.split("\t")
            if len(cols) < 16:
                continue
            row = {"case": cols[0], "params": cols[1], "pid": int(cols[2])}
            for i, k in enumerate(NUM_COLS):
                try:
                    row[k] = float(cols[3 + i])
                except ValueError:
                    row[k] = float("nan")
            row["samples"] = [float(x) for x in cols[15].split(",") if x]
            row["gc_per_rep"] = float(cols[16]) if len(cols) > 16 and cols[16] else float("nan")
            if row["case"] == "__done__":
                done = True
                row["peak_rss_mb"] = row["check"]
            rows.append(row)
    return meta, rows, done, notes


def segment_dirs(res):
    return sorted(d for d in glob.glob(os.path.join(res, "runs", "*")) if os.path.isdir(d))


def binary_of_segment(seg):
    return os.path.basename(seg).split(".")[0]


def expected_parties(binary, meta_by_pid):
    if binary == "btp":
        return [1]
    n = 3
    for meta in meta_by_pid.values():
        if "parties" in meta:
            n = int(meta["parties"])
    return list(range(n))


def check_segment(seg, binary, cases_arg):
    files = {}
    for path in glob.glob(os.path.join(seg, "CP*.tsv")):
        pid = int(re.search(r"CP(\d+)\.tsv$", path).group(1))
        files[pid] = read_tsv(path)
    metas = {pid: v[0] for pid, v in files.items()}
    parties = expected_parties(binary, metas)
    problems = []
    for pid in parties:
        if pid not in files:
            problems.append(f"CP{pid}: no result file")
        elif not files[pid][2]:
            problems.append(f"CP{pid}: no __done__ row (crashed, hung or killed)")
        for note in (files[pid][3] if pid in files else []):
            if note.startswith("# missing"):
                problems.append(f"CP{pid}: {note[2:]}")
    expected = [p[:-1] if p.endswith("$") else p for p in split_csv(cases_arg)]
    expected += [c for c in always_cases(binary) if c not in expected]
    have = {}
    for pid, (_, rows, _, _) in files.items():
        for r in rows:
            have.setdefault(r["case"], set()).add(pid)
    compute = [p for p in parties if p != 0] or parties
    missing = [c for c in expected if not (have.get(c, set()) & set(compute))]
    if missing:
        problems.append("cases without a compute-party row: " + ",".join(missing))
    return {"segment": os.path.basename(seg), "binary": binary, "parties": parties,
            "complete": not problems, "problems": problems, "cases_expected": len(expected),
            "cases_found": len([c for c in expected if c not in missing])}


def case_table(res, party="max"):
    """{case: summary dict} over every segment of a result dir. party: 'max' = slowest compute party per statistic,
    or a party id ('1', '2', ...)."""
    table = {}
    for seg in segment_dirs(res):
        binary = binary_of_segment(seg)
        per_case = {}
        for path in sorted(glob.glob(os.path.join(seg, "CP*.tsv"))):
            meta, rows, done, _ = read_tsv(path)
            for r in rows:
                if r["case"] == "__done__":
                    continue
                per_case.setdefault(r["case"], []).append(r)
        for case, rows in per_case.items():
            compute = [r for r in rows if r["pid"] != 0] or rows
            if party == "max":
                chosen = compute
            else:
                chosen = [r for r in compute if r["pid"] == int(party)] or compute
            slowest = max(chosen, key=lambda r: r["median_s"])
            entry = {
                "case": case, "binary": binary, "segment": os.path.basename(seg), "params": slowest["params"],
                "party": "max" if party == "max" and len(chosen) > 1 else f"CP{chosen[0]['pid']}" if len(chosen) == 1 else f"CP{slowest['pid']}",
                "parties": ",".join(f"CP{r['pid']}" for r in compute),
                "median_s": max(r["median_s"] for r in chosen),
                "min_s": max(r["min_s"] for r in chosen),
                # spread of the party whose median is reported (a no-op party's sub-microsecond jitter is not noise)
                "mad_rel": slowest["mad_rel"],
                "reps": int(min(r["reps"] for r in chosen)),
                "warm_s": max(r["warm_s"] for r in chosen),
                "check": max((r["check"] for r in chosen), key=abs),
                "bytes_per_rep": int(max(r["bytes_per_rep"] for r in chosen)),
                "wall_s": max(r["wall_s"] for r in chosen),
                "samples": slowest["samples"],
                "cp0_median_s": next((r["median_s"] for r in rows if r["pid"] == 0), float("nan")),
                "gc_per_rep": max(r.get("gc_per_rep", float("nan")) for r in chosen),
            }
            table[case] = entry
    return table


# ------------------------------------------------------------------------------------------------------------------
# Cache key, manifest
# ------------------------------------------------------------------------------------------------------------------

def sha256_file(path, h=None):
    h = h or hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h


def tree_files(root):
    out = []
    for d, _, fs in os.walk(root, followlinks=True):
        for f in fs:
            out.append(os.path.join(d, f))
    return sorted(out)


def stdlib_hash():
    h = hashlib.sha256()
    root = os.path.join(REPO, "stdlib")
    for p in tree_files(root):
        h.update(os.path.relpath(p, root).encode())
        sha256_file(p, h)
    return h.hexdigest()


def toolchain_files():
    home = os.path.expanduser("~")
    cands = [os.path.join(home, ".sequre", "bin", "codon"), os.path.join(home, ".sequre", "bin", "sequre"),
             os.path.join(home, ".sequre", "lib", "codon", "libcodonc.so"),
             os.path.join(home, ".sequre", "lib", "codon", "libcodonrt.so")]
    cands += sorted(glob.glob(os.path.join(home, ".sequre", "lib", "codon", "plugins", "sequre", "build", "*.so")))
    return [p for p in cands if os.path.exists(p)]


def cache_key(binary):
    h = hashlib.sha256()
    h.update(f"bench-cache-v1|{binary}|-release".encode())
    for p in sorted(glob.glob(os.path.join(BENCH_DIR, "*.codon"))) + [os.path.join(REPO, "applications", "king.codon")]:
        h.update(os.path.relpath(p, REPO).encode())
        sha256_file(p, h)
    h.update(stdlib_hash().encode())
    for p in toolchain_files():
        st = os.stat(p)
        h.update(f"{p}|{st.st_size}|{int(st.st_mtime)}".encode())
    return h.hexdigest()[:16]


def git(*args):
    try:
        return subprocess.run(["git", "-C", REPO] + list(args), capture_output=True, text=True, check=True).stdout
    except Exception:
        return ""


def git_state():
    sha = git("rev-parse", "HEAD").strip()
    status = [l for l in git("status", "--porcelain").splitlines() if l.strip()]
    h = hashlib.sha256()
    h.update(git("diff", "HEAD", "--binary").encode())
    for p in git("ls-files", "--others", "--exclude-standard").splitlines():
        full = os.path.join(REPO, p)
        if os.path.isfile(full):
            h.update(p.encode())
            sha256_file(full, h)
    return {
        "sha": sha, "short": sha[:7], "branch": git("rev-parse", "--abbrev-ref", "HEAD").strip(),
        "dirty": bool(status), "status": status, "diff_sha256": h.hexdigest(),
        "stdlib_diff_files": [l[3:] for l in status if l[3:].startswith("stdlib/")],
    }


def read_first(path, default=""):
    try:
        with open(path) as f:
            return f.read().strip()
    except OSError:
        return default


def parse_cpu_list(spec):
    """'0-5,8,10-11' -> [0, 1, 2, 3, 4, 5, 8, 10, 11]."""
    out = []
    for part in spec.split(","):
        part = part.strip()
        if part:
            lo, _, hi = part.partition("-")
            out.extend(range(int(lo), int(hi or lo) + 1))
    return out


def format_cpu_list(cpus):
    """[0, 1, 2, 5] -> '0-2,5'."""
    cpus, out, i = sorted(set(cpus)), [], 0
    while i < len(cpus):
        j = i
        while j + 1 < len(cpus) and cpus[j + 1] == cpus[j] + 1:
            j += 1
        out.append(str(cpus[i]) if i == j else f"{cpus[i]}-{cpus[j]}")
        i = j + 1
    return ",".join(out)


def one_thread_per_core(cpus):
    """The CPUs of `cpus` minus SMT siblings: per physical core, the lowest-numbered of its threads in `cpus`."""
    pool, keep = set(cpus), []
    for c in sorted(pool):
        siblings = read_first(f"/sys/devices/system/cpu/cpu{c}/topology/thread_siblings_list")
        if not siblings:
            return sorted(pool)  # no topology information: keep every CPU
        if min(s for s in parse_cpu_list(siblings) + [c] if s in pool) == c:
            keep.append(c)
    return keep


def resolve_cpus(cpus_arg, pin_arg, parties):
    """Resolves --cpus and --pin ('auto', 'none' or explicit lists) into (cpu list or '', pin spec or 'none')."""
    try:
        allowed = sorted(os.sched_getaffinity(0))
    except AttributeError:  # not Linux
        allowed = list(range(os.cpu_count() or 1))
    try:
        cpus = (one_thread_per_core(allowed) if cpus_arg == "auto" else
                [] if cpus_arg in ("none", "") else parse_cpu_list(cpus_arg))
        if pin_arg != "auto":
            for block in (pin_arg.split(":") if pin_arg not in ("none", "") else []):
                parse_cpu_list(block)
            return format_cpu_list(cpus), pin_arg or "none"
    except ValueError:
        raise SystemExit(f"bench: bad CPU list in --cpus '{cpus_arg}' or --pin '{pin_arg}'")
    cores = one_thread_per_core(cpus or allowed)
    size = len(cores) // (parties + 1)
    if size == 0:
        print(f"bench: {len(cores)} cores are too few for {parties} parties plus a block for the system; not "
              "pinning the parties", file=sys.stderr)
        return format_cpu_list(cpus), "none"
    return format_cpu_list(cpus), ":".join(format_cpu_list(cores[k * size:(k + 1) * size]) for k in range(parties))


def machine_info():
    model = ""
    for line in read_first("/proc/cpuinfo").splitlines():
        if line.startswith("model name"):
            model = line.split(":", 1)[1].strip()
            break
    mem_kb = 0
    for line in read_first("/proc/meminfo").splitlines():
        if line.startswith("MemTotal:"):
            mem_kb = int(line.split()[1])
    cores = set()
    for p in glob.glob("/sys/devices/system/cpu/cpu[0-9]*/topology/core_id"):
        pkg = read_first(os.path.join(os.path.dirname(p), "physical_package_id"), "0")
        cores.add((pkg, read_first(p)))
    return {
        "host": platform.node(), "kernel": platform.release(), "cpu_model": model,
        "logical_cpus": os.cpu_count(), "physical_cores": len(cores) or None,
        "mem_total_gib": round(mem_kb / 1024 / 1024, 1),
        "governor": read_first("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor"),
        "boost": read_first("/sys/devices/system/cpu/cpufreq/boost"),
    }


def sequre_config(res):
    cfg = {}
    for seg in segment_dirs(res):
        for path in glob.glob(os.path.join(seg, "CP*.tsv")):
            meta = read_tsv(path)[0]
            for k in ("MPC_INT_SIZE", "NUM_THREADS", "lattiseq_threads", "parties", "reps_scale", "gc_each_rep", "gc_forced", "long_rep_s", "gc_markers_restarted"):
                if k in meta:
                    cfg.setdefault(k, meta[k])
            if "pin" in meta:
                cfg.setdefault("pinned_cpus_by_party", {})[f"CP{meta.get('pid', '?')}"] = meta["pin"]
            if "params" in meta and meta.get("binary") == "btp":
                cfg.setdefault("btp_params", meta["params"])
    m = re.search(r"^DEFAULT_PARAMS\s*=\s*(\w+)", read_first(os.path.join(REPO, "stdlib", "sequre", "lattiseq", "params.codon")), re.M)
    cfg["DEFAULT_HE_PARAMS"] = m.group(1) if m else "?"
    m = re.search(r"^MPC_INT_SIZE: Static\[int\] = (\d+)", read_first(os.path.join(REPO, "stdlib", "sequre", "settings.codon")), re.M)
    cfg.setdefault("MPC_INT_SIZE", m.group(1) if m else "?")
    return cfg


def parse_build_log(path):
    out = {}
    for line in read_first(path).splitlines():
        m = re.match(r"^\[T\] (parse|typecheck|translate|compile|llvm/opt1|llvm/opt2|llvm/gpu) = ([0-9.]+)", line)
        if m:
            out[m.group(1)] = float(m.group(2))
    return out


def load_events(res):
    events = []
    path = os.path.join(res, "events.jsonl")
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                if line.strip():
                    events.append(json.loads(line))
    return events


def write_manifest(res, label, argv):
    events = load_events(res)
    builds = [e for e in events if e.get("type") == "build"]
    runs = [e for e in events if e.get("type") == "run"]
    for b in builds:
        if b.get("log"):
            b["phases_s"] = parse_build_log(os.path.join(res, b["log"]))
    shim = os.path.expanduser("~/.sequre/lib/sequre_align64.so")
    env_events = [e for e in events if e.get("type") == "env"]
    env = env_events[-1] if env_events else {}
    manifest = {
        "label": label,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "command": argv,
        "git": git_state(),
        "stdlib_sha256": stdlib_hash(),
        "machine": machine_info(),
        "sequre": sequre_config(res),
        "environment": {
            "ld_preload_shim": env.get("ld_preload", ""),
            "shim_set": bool(env.get("ld_preload")),
            "shim_exists": os.path.exists(shim),
            "sequre_gmp_path": env.get("gmp", ""),
            "taskset_cpus": env.get("cpus", ""),
            "party_pinning": env.get("pin", ""),
            "gc_flags": env.get("gc_flags", ""),
            "gc_initial_heap_size": env.get("gc_heap", ""),
            "omp_env": {k: v for k, v in os.environ.items() if k.startswith("OMP_")},
            "reps_scale": env.get("reps_scale", "1"),
            "selection": {"cases": env.get("cases", ""), "skip": env.get("skip", "")},
            "transport": "local mode: UNIX-domain sockets, no TLS/TCP (TLS and Nagle effects are not measured)",
        },
        "builds": builds,
        "runs": runs,
        "totals": {
            "build_wall_s": round(sum(float(b.get("wall_s", 0)) for b in builds if not b.get("cached")), 2),
            "run_wall_s": round(sum(float(r.get("wall_s", 0)) for r in runs), 2),
            "complete": all(r.get("complete") in (True, "1", "true") for r in runs) if runs else False,
            "load_at_start": env.get("load", ""),
            # includes the suite's own just-finished processes (the 1-minute load average lags)
            "max_load1_incl_own": max([float(r.get("load_before", "0").split()[0]) for r in runs] +
                                      [float(r.get("load_after", "0").split()[0]) for r in runs] or [0.0]),
        },
    }
    with open(os.path.join(res, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2, sort_keys=False)
        f.write("\n")
    return manifest


# ------------------------------------------------------------------------------------------------------------------
# Formatting
# ------------------------------------------------------------------------------------------------------------------

def fmt_time(s):
    if s != s:  # nan
        return "-"
    if s >= 1.0:
        return f"{s:.3f}s"
    if s >= 1e-3:
        return f"{s * 1e3:.3f}ms"
    return f"{s * 1e6:.1f}us"


def fmt_check(c):
    if c == 0:
        return "0"
    if abs(c) >= 1e4 or abs(c) < 1e-3:
        return f"{c:.2e}"
    return f"{c:.4g}"


def write_summary(res, party="max", quiet=False):
    table = case_table(res, party)
    order = {name: i for i, (_, name, _, _) in enumerate(registry())}
    rows = sorted(table.values(), key=lambda e: order.get(e["case"], 10 ** 6))
    path = os.path.join(res, "summary.tsv")
    with open(path, "w") as f:
        f.write("case\tbinary\tparams\tparty\tparties\treps\tmedian_s\tmin_s\tmad_rel\twarm_s\tcheck\tbytes_per_rep\tcp0_median_s\tgc_per_rep\n")
        for e in rows:
            f.write(f"{e['case']}\t{e['binary']}\t{e['params']}\t{e['party']}\t{e['parties']}\t{e['reps']}\t"
                    f"{e['median_s']:.6g}\t{e['min_s']:.6g}\t{e['mad_rel']:.4f}\t{e['warm_s']:.6g}\t{e['check']:.6g}\t"
                    f"{e['bytes_per_rep']}\t{e['cp0_median_s']:.6g}\t{e['gc_per_rep']:.3g}\n")
    if not quiet:
        print(f"{'case':<36} {'params':<34} {'reps':>4} {'median':>11} {'min':>11} {'mad%':>6} {'gc/rep':>6} {'check':>10}")
        for e in rows:
            print(f"{e['case']:<36} {e['params'][:34]:<34} {e['reps']:>4} {fmt_time(e['median_s']):>11} "
                  f"{fmt_time(e['min_s']):>11} {100 * e['mad_rel']:>6.1f} {e['gc_per_rep']:>6.2g} {fmt_check(e['check']):>10}")
        print(f"summary: {path} ({len(rows)} cases)")
    return rows


# ------------------------------------------------------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("list")
    p.add_argument("--cases", default="")
    p.add_argument("--skip", default="")
    p.add_argument("--binaries", default=",".join(BINARIES))
    p = sub.add_parser("plan")
    p.add_argument("--cases", default="")
    p.add_argument("--skip", default="")
    p.add_argument("--binaries", default=",".join(BINARIES))
    p = sub.add_parser("cachekey")
    p.add_argument("--binary", required=True)
    p = sub.add_parser("check")
    p.add_argument("--dir", required=True)
    p.add_argument("--binary", required=True)
    p.add_argument("--cases", default="")
    p = sub.add_parser("event")
    p.add_argument("--dir", required=True)
    p.add_argument("kv", nargs="*")
    p = sub.add_parser("manifest")
    p.add_argument("--dir", required=True)
    p.add_argument("--label", required=True)
    p.add_argument("--argv", default="")
    p = sub.add_parser("cpus")
    p.add_argument("--cpus", default="auto")
    p.add_argument("--pin", default="auto")
    p = sub.add_parser("summary")
    p.add_argument("--dir", required=True)
    p.add_argument("--party", default="max")
    p.add_argument("--quiet", action="store_true")
    a = ap.parse_args()

    if a.cmd == "list":
        bins = split_csv(a.binaries)
        pl = plan(a.cases, a.skip, bins)
        for b, name, flags, desc in registry():
            if b not in bins:
                continue
            state = ("run" if name in pl[b]["main"] else "iso" if name in pl[b]["iso"] else
                     "skip" if any(n == name for n, _ in pl[b]["skipped"]) else "-")
            if flags == "always" and (pl[b]["main"] or pl[b]["iso"]):
                state = "run"
            print(f"{state:<4} {b:<4} {name:<36} {flags:<14} {desc}")
    elif a.cmd == "plan":
        pl = plan(a.cases, a.skip, split_csv(a.binaries))
        for b in split_csv(a.binaries):
            if pl[b]["main"]:
                print(f"{b} main {','.join(n + '$' for n in pl[b]['main'])}")
            for n in pl[b]["iso"]:
                print(f"{b} iso {n}$")
            for n, flags in pl[b]["skipped"]:
                print(f"{b} skipped {n} {flags}")
    elif a.cmd == "cachekey":
        print(cache_key(a.binary))
    elif a.cmd == "check":
        res = check_segment(a.dir, a.binary, a.cases)
        print(json.dumps(res))
        sys.exit(0 if res["complete"] else 1)
    elif a.cmd == "event":
        ev = {}
        for kv in a.kv:
            k, _, v = kv.partition("=")
            if v in ("true", "false"):
                ev[k] = v == "true"
            else:
                try:
                    ev[k] = int(v) if re.fullmatch(r"-?\d+", v) else float(v) if re.fullmatch(r"-?\d+\.\d*", v) else v
                except ValueError:
                    ev[k] = v
        with open(os.path.join(a.dir, "events.jsonl"), "a") as f:
            f.write(json.dumps(ev) + "\n")
    elif a.cmd == "manifest":
        write_manifest(a.dir, a.label, a.argv)
    elif a.cmd == "cpus":
        # The parties of a local run: SEQURE_CP_COUNT, as in stdlib/sequre/constants.codon (default 3).
        cpus, pin = resolve_cpus(a.cpus, a.pin, int(os.environ.get("SEQURE_CP_COUNT", "3")))
        print(f"{cpus or '-'} {pin}")
    elif a.cmd == "summary":
        write_summary(a.dir, a.party, a.quiet)


if __name__ == "__main__":
    main()
