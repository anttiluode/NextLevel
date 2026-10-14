# NextLevel — is self-altering geometry the next memory architecture?

**Status: Gate 0 implemented and run; architectural hypothesis OPEN.**


## Gate 1 (October 14): the learned routing hypothesis failed its first pilot

**[Explore the Gate 1 interactive results](gate1.html)** · [Gate 1 protocol](GATE1_PROTOCOL.md) · [Three-seed machine-readable summary](results/gate1/pilot350/summary.json) · [Source and trainer](train_gate1.py)

Gate 1 now implements a **trainable online writer**, an actual 64-byte quantized phase memory and **query-conditioned route/order policy**, plus a separate identical-initialization frozen-route control. Trained comparison arms are a 64-byte GRU, fast-weight matrix and 32-record attention reader. Strong deterministic controls are a 64-byte exact last-write window and a **16-byte** exact indexed bit table (possible because the 128-key domain is known). Each write round-trips through 8-bit quantization.

The 350-step exploratory CPU pilot has three independent seeds, 512 held-out streams per scenario/length/seed, multiple unseen sequence lengths, and forced key rewrites. **At 64 independent writes, the learned graph-ratio head averages 82.36%, the same-initialization frozen-router head 86.26%, learned KV-style attention 77.73%, exact 64-byte 32-record window 80.53%, and exact 16-byte bit table 100%.** The frozen router wins every pilot seed. The graph's ratio observation is algebraically equivalent to directly reading the addressed phase (numerical discrepancy around 1e-6 or lower), at much greater path computation cost. **No learned-geometry gain is demonstrated.** A trained question-dependent order mixture also failed to improve on ratio/direct. These findings are *not* evidence that Transformers are obsolete.

This is an initial mechanistic training test, not a completed architecture bake-off: no full causal Transformer, no matched-FLOP allocation, fixed known keys, no online feedback after probe responses, and no natural images. It provides a working instrument for the next experimental gate. To reproduce:

```bash
python -m pip install -r requirements-gate1.txt
python -m unittest discover -s tests -v
python train_gate1.py --steps 350 --batch 64 --length 24 --eval-cases 512 --seeds 41 97 127 --device auto --out results/gate1/reproduced350
```

**Gate 0 (unchanged):** The original deterministic capacity and common-path cancellation controls follow below.

Two related October 2026 projects provide a motivating observation:
[SihtiResponseGraph](https://github.com/anttiluode/SihtiResponseGraph) writes synthetic image information into queryable conductances; [SihtiGrapImageModel](https://github.com/anttiluode/SihtiGrapImageModel) writes 64-byte phase maps and verifies exact common-path cancellation for ordered unitary probes. These are real *toy-model* measurements, not evidence that Transformers have ended.

This repository asks a falsifiable question:

> Can a fixed-budget, experience-written geometric state answer changing questions more accurately, robustly, or cheaply than comparably budgeted conventional memories **when we charge for addressing, stored bytes, reads, and forgotten facts**?

**[Experiment dashboard](https://anttiluode.github.io/NextLevel/)** · [Frozen protocol](PROTOCOL.md) · [First 36-trial receipts](results/pilot.json) · [CSV](results/pilot.csv)

## The first gate: two separate effects

1. **Positive control — measurement:** In the ideal unitary path model, two operator orders give
   `z1 = exp(i*(source + common))`, `z2 = exp(i*(common + port))`, so `z1*conj(z2) = exp(i*(source-port))` exactly. Shared path-phase nuisance cancels; there are two physical readings rather than one. This is known common-mode rejection, not a newly discovered theorem.
2. **Falsification — memory:** A 64-byte phase array has finite capacity. Distinct, independently random facts collide. Paired interrogation cannot restore a value that an overwrite has erased. The experiment compares an equally sized 64-byte key/value tape, an exact **32-byte** indexed bit table (possible because the key universe is known), and a 2-byte structured-group oracle.

## What is and isn't simulated

A stream writes bits for keys in the **known domain 0–255**, in random order, into three independently constructed stores:

| Memory | Stored bytes | What the read is allowed to do |
| --- | ---: | --- |
| Phase geometry | 64 | Public key-to-source route; launch to port 0 on an 8×8 toroidal lattice, then sample 1 or 2 complex readings |
| Bounded tape | 64 | Search last 32 two-byte `(key,value)` records; missing fact receives chance-level expected accuracy |
| Exact indexed bitset | **32** | Direct key addressing to one of 256 bits; strongest domain-specific baseline |
| Structured group oracle | **2** | Only valid when the *given* group-of-16 rule is true |
| Full ledger | `2 × facts` | Store every two-byte record, unconstrained memory growth |

The phase model retains a reference port and **63 writable phase sites**. Its public router hashes a key to a site. Two keys at the same site overwrite each other: there is no secretly stored key tag. Each site uses 8 bits to encode a one-bit fact. The ordered path contains 12 shifts, including identity/roundtrip steps as required, so all measurements end at a fixed port. The source coordinate is *privileged question-derived address information*. We do not claim the experiment learned how to route, where to write, or how to represent a natural image.

`independent_bits`: every key gets its own random bit. `shared_group_bits`: 16 adjacent keys share one random group bit, and the geometric router is **given** that rule. `broken_group_rule`: the router assumes group structure but facts are independent. Questions cover all previously written keys; no training occurs. An optional read-time attack changes all intermediate *path visit* phases while leaving endpoints unchanged.

## Frozen first results

**36 independent trials per scenario/count, seed 41**; evaluation is a deterministic, no-training simulator, not a model trained on 36 examples. Values below are query accuracy; full receipts include standard error per trial and attack cases.

| Facts in stream | Independent: phase ratio | Independent: 64B tape | Known-key exact bitset | Matched-group phase | Broken-group phase |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 8 | 97.2% | 100.0% | 100.0% | 100.0% | 88.2% |
| 32 | 91.1% | 100.0% | 100.0% | 100.0% | 71.4% |
| 64 | 83.9% | 75.0% | 100.0% | 100.0% | 62.2% |
| 128 | 74.4% | 62.5% | 100.0% | 100.0% | 54.6% |
| 256 | 62.5% | 56.2% | 100.0% | 100.0% | 54.2% |

The graph-style phase store outperforms the deliberately basic bounded tape at high independent load but loses to exact **32-byte** bit indexing. Its excellent group performance evaporates when the assumed grouping is false. Both results matter: the budget *and the match between stored structure and future questions* determine usefulness.

The paired ratio remains unchanged under added intermediate-path phase noise and equals privileged direct-source phase in this ideal model. Single-wave accuracy approaches chance under the same nuisance. The paired ratio is a deterministic function of **both raw signals**; a sufficiently expressive two-signal reader can calculate it too. We do not assert extra information was created by the ratio.

## Reproduce with any Python 3.10+ (no dependencies)

```bash
python -m unittest discover -s tests -v
python run.py --seed 41 --trials 36 --out results/pilot.json
python run.py --seed 97 --trials 100 --out results/replication.json
```

The generated receipts record all points, conditions, runs and memory budgets; outputs include `.json` and `.csv`. A smoke run is included in GitHub Actions.

## What would justify an architectural claim?

A positive architectural gate needs a **trained online writer and learned query-dependent reader**, independent held-out tasks, honest matched memory/inference budgets, controlled access to past information, and comparison with learned recurrent fast weights, a small attention/KV model, simple indexing, and retrieval. Neither a handcrafted interferometer nor an oracle group rule satisfies that gate. Compare long-horizon update interference, counterfactual fact changes, novel query combinations, robustness to incorrect structure, throughput and GPU memory.

Gemini's October 14 proposal mixes a testable ingredient with a much larger unsupported extrapolation: the fact that a 64-byte store does not grow with time is not proof it can retain arbitrary information written during that time. For `N` independent unbiased binary facts, exact recall of all N requires at least N bits in the worst case, unless the information is accessible elsewhere (shared prior, external memory, recomputation, extra reads, or a weaker task). The *bytes in learned shared weights* can be amortized over many examples, but must still be reported.

## Related work, not a novelty claim

- [Titans: Learning to Memorize at Test Time (NeurIPS 2025)](https://proceedings.neurips.cc/paper_files/paper/2025/hash/a4ca07aa108036f80cbb5b82285fd4b1-Abstract-Conference.html)
- [ATLAS: Learning to Optimally Memorize the Context at Test Time (ICML 2026)](https://proceedings.mlr.press/v306/behrouz26b.html)
- [TNT: Improving Chunkwise Training for Test-Time Memorization (ICLR 2026)](https://proceedings.iclr.cc/paper_files/paper/2026/hash/bdb30687f1c2255c29b11b0b45204ebe-Abstract-Conference.html)
- [An Empirical Study of Mamba-based Language Models (2024)](https://arxiv.org/abs/2406.07887)

**The goal is not to pronounce the Transformer dead; it is to find exactly what task a new memory mechanism can do better, and at what price.**
