# NextLevel, frozen Gate 0 protocol — 2026-10-14

## Research claim / null

**Strong claim under scrutiny:** a self-altering fixed-size phase or conductance geometry, interrogated through ordered non-commuting operations, can *replace* an indefinitely growing token/fact tape while retaining usable arbitrary history.

**Null:** a bounded geometry has a finite information capacity; useful phase cancellation at readout does not circumvent irreversible write collisions. Specialized geometry wins only where shared structure, constraints, or a decoder prior make the query answers compressible.

## Outcomes

- Fact retrieval mean accuracy over every written key, measured at each stream length `N ∈ {8,16,32,64,96,128,192,256}`.
- All memories process the same sequence of distinct `(key,bit)` writes. Keys are uniformly sampled from the shared public key domain `0..255`. Independent bits are Bernoulli 0.5.
- A group experiment samples one independent bit per 16-key group; its router is explicitly given the 16-key grouping. A misaligned control uses the same router but independent facts.
- Mean over 36 seeded independent trials per point; trial standard errors are reported. Multiple query rows within a trial are correlated and **not** counted as independent seeds.
- Read-out arms: phase-to-port one wave, reversed wave, raw pair as ratio (algebraically equivalent), complex order difference (not demodulated), direct-addressed source phase. Only the ratio is an exact common-mode cancellation. Both clean and perturbed common path conditions are evaluated on identical data trials.

## Unit cost and access contracts

- Phase geometry: 64 actual `uint8` cells, including fixed 0-phase readout port at index 0, 63 writable sites. Per-key router `1+((73*k+19) mod 63)` is public and free, but not learned. For structured memory router `1+floor(k/16)` is also given, not learned.
- Each written bit becomes phase π/2 for 1 or 3π/2 for 0, quantized exactly as uint8 64 or 192. Single writes overwrite a collided cell. There is **no identity tag**, and overwritten information cannot be recovered by querying a path.
- The path is a query-selected 12-step walk with nearest-neighbor toroidal moves, identity steps and two-step excursions to fix path length. A read is a unit-modulus complex endpoint value: `z1=exp(i sum(phase[p0:p11]))` or `z2=exp(i sum(phase[p1:p12]))`. Two paired shots cost **24 phase/shift steps total**, twice a one-way shot, and report 2 complex numbers. No extra source-phase measurements are permitted for the paired arm.
- In the nuisance attack, independent additional phase is added to visits `p1..p11`, **identically for both orderings**. The cancellation is therefore expected algebraically; non-shared noise, amplitude attenuation, and physical phase disorder are not modeled.
- Bounded tape: 32 `(uint8 key, uint8 value)` records = 64 bytes, last-write-wins recency. Missing answer is scored at exact chance probability 0.5, not by an arbitrary guessed bit. It is **not** a Transformer model or an optimized KV cache.
- Domain-specific exact bit table: 256 indexable bits = **32 bytes**; for these known fixed keys no key tags are needed. Serves as strongest baseline and a reminder to compare proper encoding, not just a generic tape.
- Structurally informed oracle: 16 group bits = **2 bytes** when and only when every key in a 16-key group really shares the same value. Represents shared prior supplied as free codebook/rule.
- Unlimited ledger: N `(uint8 key, uint8 value)` records = 2N bytes. No masking of its increasing cost.

## Gates

1. **G0-A (must PASS):** cancellation identity within 1e-10 and invariance under perturbation of the shared intermediate visits, including saved quantized memories. Unit tests cover every launch site.
2. **G0-B (must PASS):** deliberately create collisions and confirm erased facts are not magically recoverable; 32-byte indexed bitset remains exact on all 256 independent keyed facts, including rewrites.
3. **G0-C (expected conditional):** phase geometry accuracy declines as independent fact load increases; group prior wins under correct structure and fails if that prior is wrong. Never rebrand a hard-coded router as learned geometry.
4. **G0-D (interpretation):** paired ratio cannot beat a reader given both complex measurements under an information-theoretically optimal decoder; measured differences between learned heads would reflect inductive bias, training, or computation.
5. **G1 (NOT BUILT):** learned memory writer and route policy, with sequence-to-sequence or interactive reasoning under a fixed inference-memory budget and matched trained baselines. A claim about Transformers requires G1 or stronger.

## Threats to validity

Synthetic independent/grouped key-bit data, engineered phases, privileged key-address routing, binary targets, fixed known key space, chosen unitary/noise model, inability to assess unobserved-world modeling, small CPU simulator, and a simplistic bounded tape. No pretrained language model is involved. A 64-byte map can encode at most 512 arbitrary bits, **not zero bits**; the observed loss below that limit is caused by our intentionally sparse colliding addressing, not a general capacity theorem for all 64-byte memories. Unlimited independent history is impossible at fixed total bits, but a better 64-byte encoder could preserve *all 256 bits in our exact task*.

The numerical results cannot establish that a particular learned geometry is superior, that brain dendrites use phase interferometry, or that attention is obsolete.
