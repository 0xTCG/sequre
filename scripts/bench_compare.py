#!/usr/bin/env python3
"""
Compare two scripts/bench.sh result directories case by case, or calibrate the noise from two runs of the same code.
Standard library only.

    bench_compare.py BASE CAND [--party max|1|2] [--noise FILE|none] [--min-pct 5] [--k 2] [--cases P1,P2]
                     [--changed] [--fail-on-regression]
    bench_compare.py --calibrate RUN1 RUN2 [--out FILE]

Per case, both sides use the same party rule (default 'max': per statistic, the slowest compute party; CP0, the
dealer, is never used; solo cases exist only on CP1). For each case:

    dmed, dmin  = relative change of the median and of the minimum, candidate vs base (%)
    se          = standard error of a median, from the samples' MAD: 1.86 * MAD / median / sqrt(reps), both sides
                  combined (sqrt(se_b^2 + se_c^2))
    noise       = max(calibrated run-to-run noise of the case (--noise file), se)
    threshold   = max(--min-pct, --k * noise)
    REGRESSION  when dmed > threshold and dmin > threshold; FASTER when both < -threshold; ~ otherwise.

The witness column (`check`: an error against plaintext, a mismatch count, a fraction...) is compared too; a change
of more than 30x, or between zero and non-zero, is listed as CHECK (a change in results, not in speed). A change of
at least 0.5 garbage collections per repetition is shown in parentheses (allocation volume; informational only:
collection counts also vary a little between runs of identical code).

--calibrate writes, per case, the run-to-run change of the median and of the minimum between two runs of the same
code and noise_pct = max(|dmed|, |dmin|, se); compare reads it with --noise (default: results/bench/noise.tsv when
present). Exit status: 0, or 1 with --fail-on-regression when a case regressed; 2 on bad input.
"""
import argparse
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bench_tool  # noqa: E402

DEFAULT_NOISE = os.path.join(bench_tool.REPO, "results", "bench", "noise.tsv")


def se_rel(e):
    reps = max(int(e["reps"]), 1)
    return 1.86 * e["mad_rel"] / math.sqrt(reps)


def pct(new, old):
    if old <= 0 or new != new or old != old:
        return float("nan")
    return 100.0 * (new - old) / old


def check_changed(c1, c2):
    if c1 != c1 or c2 != c2:
        return (c1 != c1) != (c2 != c2)
    a, b = abs(c1), abs(c2)
    if a == b:
        return False
    if (a == 0) != (b == 0):
        return max(a, b) > 1e-6
    hi, lo = max(a, b), min(a, b)
    return hi > 1e-6 and hi / lo > 30


def load_noise(path):
    noise = {}
    if not path or path == "none" or not os.path.exists(path):
        return noise
    with open(path) as f:
        header = f.readline().rstrip("\n").split("\t")
        idx = header.index("noise_pct")
        for line in f:
            cols = line.rstrip("\n").split("\t")
            if len(cols) > idx:
                noise[cols[0]] = float(cols[idx]) / 100.0
    return noise


def load_manifest(d):
    try:
        with open(os.path.join(d, "manifest.json")) as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def in_selection(case, patterns):
    return not patterns or any(bench_tool.matches(case, p) for p in patterns)


def ordered(cases):
    order = {name: i for i, (_, name, _, _) in enumerate(bench_tool.registry())}
    return sorted(cases, key=lambda c: (order.get(c, 10 ** 6), c))


def calibrate(run1, run2, out):
    t1, t2 = bench_tool.case_table(run1), bench_tool.case_table(run2)
    common = ordered(set(t1) & set(t2))
    rows = []
    for c in common:
        a, b = t1[c], t2[c]
        dmed = pct(b["median_s"], a["median_s"])
        dmin = pct(b["min_s"], a["min_s"])
        se = 100.0 * math.sqrt(se_rel(a) ** 2 + se_rel(b) ** 2)
        noise = max(abs(dmed), abs(dmin), se)
        rows.append((c, a, b, dmed, dmin, se, noise))
    if out:
        with open(out, "w") as f:
            f.write("case\tmed1_s\tmed2_s\tdmed_pct\tmin1_s\tmin2_s\tdmin_pct\tmad1_pct\tmad2_pct\tse_pct\tnoise_pct\treps\n")
            for c, a, b, dmed, dmin, se, noise in rows:
                f.write(f"{c}\t{a['median_s']:.6g}\t{b['median_s']:.6g}\t{dmed:.2f}\t{a['min_s']:.6g}\t{b['min_s']:.6g}\t"
                        f"{dmin:.2f}\t{100 * a['mad_rel']:.2f}\t{100 * b['mad_rel']:.2f}\t{se:.2f}\t{noise:.2f}\t"
                        f"{min(a['reps'], b['reps'])}\n")
    print(f"{'case':<36} {'median run1':>12} {'run2':>12} {'dmed%':>7} {'min run1':>12} {'run2':>12} {'dmin%':>7} "
          f"{'mad%':>6} {'se%':>5} {'noise%':>6}")
    for c, a, b, dmed, dmin, se, noise in rows:
        print(f"{c:<36} {bench_tool.fmt_time(a['median_s']):>12} {bench_tool.fmt_time(b['median_s']):>12} {dmed:>+7.1f} "
              f"{bench_tool.fmt_time(a['min_s']):>12} {bench_tool.fmt_time(b['min_s']):>12} {dmin:>+7.1f} "
              f"{100 * max(a['mad_rel'], b['mad_rel']):>6.1f} {se:>5.1f} {noise:>6.1f}")
    noises = sorted(r[6] for r in rows)
    if noises:
        med = noises[len(noises) // 2]
        print(f"\n{len(rows)} cases; noise% median {med:.1f}, max {noises[-1]:.1f}; "
              f"{sum(1 for n in noises if n <= 2.5)} cases <= 2.5% (detection threshold = the 5% floor), "
              f"{sum(1 for n in noises if n > 5)} cases > 5% (threshold 2x noise > 10%)")
    only = sorted(set(t1) ^ set(t2))
    if only:
        print("only in one run: " + ", ".join(only))
    if out:
        print(f"noise file: {out}")


# Run settings that make two runs' timings incomparable when they differ: (manifest section, key).
SETTINGS = [("machine", "host"), ("machine", "cpu_model"), ("environment", "taskset_cpus"),
            ("environment", "party_pinning"), ("environment", "gc_flags"), ("environment", "gc_initial_heap_size"),
            ("environment", "reps_scale"), ("environment", "selection"), ("sequre", "parties"),
            ("sequre", "NUM_THREADS"), ("sequre", "lattiseq_threads"), ("sequre", "MPC_INT_SIZE"),
            ("sequre", "DEFAULT_HE_PARAMS"), ("sequre", "harness_version"), ("sequre", "gc_markers_restarted"),
            ("sequre", "gc_parallel_by_party")]


def settings_differences(mb, mc):
    """The SETTINGS that differ between two manifests, as 'key (base vs cand)' strings."""
    if not mb or not mc:
        return []
    out = []
    for section, key in SETTINGS:
        b, c = mb.get(section, {}).get(key), mc.get(section, {}).get(key)
        if b != c:
            out.append(f"{key} ({b} vs {c})")
    return out


def compare(base, cand, args):
    tb = bench_tool.case_table(base, args.party)
    tc = bench_tool.case_table(cand, args.party)
    patterns = bench_tool.split_csv(args.cases)
    noise_path = args.noise if args.noise else (DEFAULT_NOISE if os.path.exists(DEFAULT_NOISE) else "")
    noise = load_noise(noise_path)
    common = ordered(c for c in set(tb) & set(tc) if in_selection(c, patterns))
    regressions, faster, checks = [], [], []
    lines = []
    for c in common:
        b, n = tb[c], tc[c]
        dmed = pct(n["median_s"], b["median_s"])
        dmin = pct(n["min_s"], b["min_s"])
        se = math.sqrt(se_rel(b) ** 2 + se_rel(n) ** 2)
        nz = 100.0 * max(noise.get(c, 0.0), se)
        thr = max(args.min_pct, args.k * nz)
        if dmed > thr and dmin > thr:
            verdict = "REGRESSION"
            regressions.append(c)
        elif dmed < -thr and dmin < -thr:
            verdict = "FASTER"
            faster.append(c)
        else:
            verdict = "~"
        chk = ""
        if check_changed(b["check"], n["check"]):
            chk = f"CHECK {bench_tool.fmt_check(b['check'])} -> {bench_tool.fmt_check(n['check'])}"
            checks.append(c)
        note = chk
        gcb, gcc = b.get("gc_per_rep", float("nan")), n.get("gc_per_rep", float("nan"))
        if gcb == gcb and gcc == gcc and abs(gcc - gcb) >= 0.5:
            # informational: collection counts also vary between runs of identical code (small MPC heaps)
            note = (note + " " if note else "") + f"(gc/rep {gcb:.2g} -> {gcc:.2g})"
        if args.changed and verdict == "~" and not chk:
            continue
        lines.append(f"{c:<36} {bench_tool.fmt_time(b['median_s']):>11} {bench_tool.fmt_time(n['median_s']):>11} {dmed:>+7.1f} "
                     f"{bench_tool.fmt_time(b['min_s']):>11} {bench_tool.fmt_time(n['min_s']):>11} {dmin:>+7.1f} "
                     f"{nz:>6.1f} {thr:>5.1f}  {verdict:<10} {note}")
    print(f"base: {base}\ncand: {cand}\nparty rule: {args.party}; noise: {noise_path or 'within-run only'}; "
          f"threshold = max({args.min_pct:g}%, {args.k:g} x noise)\n")
    print(f"{'case':<36} {'base med':>11} {'cand med':>11} {'dmed%':>7} {'base min':>11} {'cand min':>11} {'dmin%':>7} "
          f"{'noise%':>6} {'thr%':>5}  verdict")
    for line in lines:
        print(line)

    only_b = [c for c in ordered(set(tb) - set(tc)) if in_selection(c, patterns)]
    only_c = [c for c in ordered(set(tc) - set(tb)) if in_selection(c, patterns)]
    if only_b:
        print("\nonly in base: " + ", ".join(only_b))
    if only_c:
        print("only in candidate: " + ", ".join(only_c))

    mb, mc = load_manifest(base), load_manifest(cand)
    builds_b = {e.get("binary"): e for e in mb.get("builds", [])}
    builds_c = {e.get("binary"): e for e in mc.get("builds", [])}
    if builds_b or builds_c:
        print("\nbuild time (s)            base        cand")
        for binary in bench_tool.BINARIES:
            if binary in builds_b or binary in builds_c:
                def show(e):
                    if not e:
                        return "-"
                    return f"{float(e.get('wall_s', 0)):.1f}{' (cached)' if e.get('cached') else ''}"
                print(f"  bench_{binary:<18} {show(builds_b.get(binary)):>12} {show(builds_c.get(binary)):>12}")
    for name, m in (("base", mb), ("cand", mc)):
        if m:
            t = m.get("totals", {})
            g = m.get("git", {})
            print(f"{name}: git {g.get('short', '?')}{' dirty ' + g.get('diff_sha256', '')[:12] if g.get('dirty') else ''}, "
                  f"run {t.get('run_wall_s', '?')}s, load at start {t.get('load_at_start', '?')}, "
                  f"complete {t.get('complete', '?')}, gc {m.get('environment', {}).get('gc_flags', '?') or 'adaptive'}")

    differ = settings_differences(mb, mc)
    if differ:
        print("WARNING: the runs differ in " + "; ".join(differ) + "; their timings may not be comparable")

    print(f"\n{len(common)} cases compared: {len(regressions)} regression(s), {len(faster)} faster, "
          f"{len(checks)} witness change(s)")
    if regressions:
        print("REGRESSIONS: " + ", ".join(regressions))
    if checks:
        print("WITNESS CHANGES: " + ", ".join(checks))
    return 1 if (args.fail_on_regression and regressions) else 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dirs", nargs=2, metavar=("BASE", "CAND"))
    ap.add_argument("--calibrate", action="store_true", help="noise calibration from two runs of the same code")
    ap.add_argument("--out", default="", help="--calibrate: noise file to write (e.g. results/bench/noise.tsv)")
    ap.add_argument("--party", default="max", help="max (slowest compute party, default) or a party id: 1, 2")
    ap.add_argument("--noise", default="", help="calibrated noise file, or 'none' (default: results/bench/noise.tsv if present)")
    ap.add_argument("--min-pct", type=float, default=5.0, help="threshold floor in %% (default 5)")
    ap.add_argument("--k", type=float, default=2.0, help="threshold = max(min-pct, k x noise) (default 2)")
    ap.add_argument("--cases", default="", help="compare only cases matching these prefixes")
    ap.add_argument("--changed", action="store_true", help="list only regressions, improvements and witness changes")
    ap.add_argument("--fail-on-regression", action="store_true", help="exit 1 when a case regressed")
    args = ap.parse_args()
    for d in args.dirs:
        if not os.path.isdir(os.path.join(d, "runs")):
            print(f"bench_compare: {d} is not a bench result directory (no runs/)", file=sys.stderr)
            sys.exit(2)
    if args.calibrate:
        calibrate(args.dirs[0], args.dirs[1], args.out)
        sys.exit(0)
    sys.exit(compare(args.dirs[0], args.dirs[1], args))


if __name__ == "__main__":
    main()
