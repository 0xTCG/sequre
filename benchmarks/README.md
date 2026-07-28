# Benchmarks

Cross-framework benchmarks that compare Sequre against other secure-computation
frameworks on the same workload, same inputs, same metrics.

These are distinct from the in-tree performance tests:

- [`tests/perf_bench.codon`](../tests/perf_bench.codon) profiles Sequre's own
  layers bottom-up (CPU primitives, HE ops, MPC ops, applications) to catch
  regressions.
- [`tests/benchmark.codon`](../tests/benchmark.codon) times Sequre's
  application pipelines.
- **This directory** puts Sequre next to other frameworks.

Each subdirectory is self-contained: a shared spec, one runner per framework, a
report generator, and a README stating what is and is not comparable.

| Benchmark | What it compares |
|---|---|
| [`activations/`](activations/) | `exp`, `sigmoid`, `tanh`, `relu` in Sequre+Decor, CrypTen and MPyC — latency, throughput, communication and accuracy |

## Adding one

Follow the shape of `activations/`:

1. `spec.py` — inputs, sizes, repetition counts. One source of truth that every
   runner reads, so no runner can quietly benchmark something else.
2. `common.py` — the result record and its JSONL serialisation, shared by the
   Python runners. Non-Python runners emit the same schema.
3. One runner per framework, in its own subdirectory, using **that framework's
   own library implementation** wherever one exists. Where it does not, say so
   in the file and in the README, and implement the standard algorithm rather
   than a favourable one.
4. `report.py` — reads `results/*.jsonl`, verifies every file was produced
   against the same spec, and writes the comparison table.
5. `README.md` — the results, and a caveats section covering differing threat
   models, precision, and anything else that makes a cell not strictly
   comparable.

Record a correctness witness (error against a plaintext reference) in the same
row as every timing. A speedup that breaks the math should be visible without
having to look anywhere else.
