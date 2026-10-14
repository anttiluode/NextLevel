# Gate 1 — online learned phase memory and query policy

**Implementation / exploratory pilot: October 14, 2026.** This protocol was documented after initial implementation and small pilot runs; **not preregistered**. The contained 350-step runs test wiring and disclose counterexamples. A fixed 1,200-step, three-seed follow-up is proposed but has not been run here.

## Question and success criterion

Can a geometry written by a sequence of observations, interrogated by a learned query-dependent probe, recover last-write key/value facts more accurately than **its own router-frozen counterpart** and a strong conventional memory at the *same inference memory cost*? Distinguish readout benefits from routing benefits, and always compare to the **16-byte exact indexed memory** available because keys are from a known 128-element universe.

Strong Gate 1 PASS requires a replicated advantage from **learning the routing policy** over the bit-for-bit identical initial frozen policy; it must also beat a sensible trained baseline at 64 bytes, and its accuracy/compute tradeoff must survive a direct-site readout baseline. Otherwise record FAILED/UNRESOLVED.

## Task

- Keys: integers 0..127, shared across train and test. This measures held-out **streams**, not unseen keys.
- Online observations: sequence of `(key, bit)` pairs. Each read asks about one previously written key; target is the **most recently written value for that key**.
- Training alternates independent write bits and within-episode grouped bits (one fixed shared bit per group of eight keys), length 24 and fresh streams every optimization step.
- Held-out tests: independent, grouped8, independent renamed broken_group (a separate seed; *not* a stronger adversarial split), forced last-write reversal, and grouped16 OOD structure; lengths 16, 32, 64 and 96; 512 episodes per scenario/count/model/seed.
- Pilot seeds 41/97/127, 350 AdamW optimization steps, batch64, `lr=0.002`, CPU; all models get the same generated training batches and disjoint held-out RNG streams. No early stopping / adaptive selection of seed or step.
- Rewrite probe picks a previously written key, changes its bit at the final step and queries that key.

## Trained models and controls

| Memory | Charged per-stream payload | Shared trainable mechanism / access |
|---|---:|---|
| PhaseGraph learned | 64 bytes uint8 | 8x8 phase state; key-conditioned 63-site writer; separate learned correction to query launch site and learned AB/BA order mixture; two complex, fixed-port wave measurements, ratio/pair/single/order-policy heads |
| PhaseGraph frozen-router | 64 bytes uint8 | **Same initial weights**, then freezes writer/read route tables; trains remaining phase code and readout heads |
| Quantized GRU | 64 bytes uint8 | GRUCell writes 64 signed quantized activation coordinates, learned key-conditioned reader |
| Low-rank fast weights | 64 bytes uint8 | 8x8 signed quantized matrix updated with learned key/value outer products, learned key readout |
| Learned window attention | 64 reserved bytes | Last 32 raw `(uint8 key, uint8 bit)` events, learned key embeddings and attention over up to 32 events |
| Exact window last-match | 64 reserved bytes | Same 32 records, exact latest-match lookup, missing answer scored as 0.5 expected accuracy |
| Exact known-key indexed memory | **16 bytes** | Exact 128-bit table with public direct index; handles overwrites; no learned checkpoint |

Learned models differ in parameter count and compute. Gate 1 does **not** claim FLOP or parameter matching. Read/compute must be included before architectural comparison.

## Geometry and the probe

Each key has a learned 63-site writer logit vector. The forward update selects the highest-logit site using a straight-through hard argmax and overwrites one byte with a learned phase code for bit 0/1. After **every** write, state is STE-quantized to true 8-bit values. Read policy has a separately learnable key-specific correction to site logits. The query also controls learned weights over the AB and BA probe orders.

Each of 63 candidate sites has a fixed 12-step toroidal route to physical port 0, inherited from Gate 0. End-port wave `z_AB` sums phases at the first 12 visits and `z_BA` sums phases at the final 12. A hard query site chooses one route. The ratio `z_AB*conj(z_BA)` is **algebraically the phase of the query's source site relative to the port** (which is fixed to phase 0). This is an identity, not automatically novel computation. The `direct_addressed` reader reads that source phase directly: it is a stronger interface and a crucial information-equivalence control. The paired raw-response reader can reconstruct the ratio, given sufficient readout capability.

**Compute:** the current PyTorch implementation computes both responses for all 63 source sites then selects one. Actual implementation compute therefore includes **63 candidate paths × 12 steps × 2 orders**, even though an optimized single selected path would need only 24 phase/shift steps. The latter still costs more than one direct memory read. The policy is **question-conditioned but not feedback-adaptive**: no second probe selection depends on the first observed response.

At training time, straight-through softmax supplies proxy gradients for the hard selected route. Hard probabilities can remain almost uniform even if the discrete argmax changes; report entropy and changed top-site count rather than declaring meaningful routing learned.

## Frozen preliminary readouts and nulls

- Ratio vs direct-site equality must hold within float arithmetic for hard routes; measured residual in JSON.
- Query-independent route is an intentionally blind ablation; success against it alone does **not** establish geometry superiority.
- Route-frozen same-initialization is the crucial geometry-learning control.
- Any exact indexed-table advantage is substantive; do not dismiss it because it uses an explicit public key universe.
- Evaluate changes in site assignment and order preferences; simply having trainable parameters is not evidence they learned a beneficial policy.
- No prompt-level reasoning, images, unobserved geometry, train-time tool use, actual Transformer, novel physics, external long-term memory, or biological dendrite claims.
- Pilot compares learned KV-style window attention, **not a full causal Transformer**. A true Transformer/SSM matched-compute study is still open.

## Reproduction

```bash
python -m pip install -r requirements-gate1.txt
python -m unittest discover -s tests -v
python train_gate1.py --smoke --device cpu
python train_gate1.py --steps 350 --batch 64 --length 24 --eval-cases 512 --seeds 41 97 127 --device cpu --out results/gate1/reproduced350
# recommended larger exploratory follow-up; not yet run:
python train_gate1.py --steps 1200 --seeds 41 97 127 --device auto --checkpoints --out results/gate1/long1200
```

Pilot receipts: `results/gate1/pilot350/seed{41,97,127}.json`; summary also reports per-seed and mean accuracies. Models are **reinitialized at every seed** and trained independently. Results do not include saved checkpoints; running with `--checkpoints` preserves weights, which should be inspected in any stronger follow-up.

## Threats and next gate

Random collisions in a sparse 63-site store, an engineered port-phase cancellation identity, known repeated keys, training on the same key alphabet, STE bias, full raw token access for window baseline versus graph phase-path access, different optimizer conditioning/model sizes, limited 350-step training, synthetic bit classification, and OOD task definition limit any inference. A new attempt should use learned, sequence-observation-dependent geometry rather than only a fixed key-to-slot embedding, control writes/queries at equal full compute, test truly unseen key encodings, and include a full causal attention/KV baseline.
