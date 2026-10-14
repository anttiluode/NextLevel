"""Standard-library reference experiment; no training or Transformer simulation.

Every method sees the same streaming key/bit writes and the same queries.
Key space is FIXED AND PUBLIC (0..255). This deliberately makes an optimal
32-byte indexed bit table possible and exposes the cost of phase geometry.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import cmath
import math
import random
from typing import Iterable

KEY_SPACE = 256
GRID_SIDE = 8
PHASE_BYTES = 64
PORT = 0
TAPE_BYTES = 64
TAPE_RECORD_BYTES = 2
BITSET_BYTES = KEY_SPACE // 8
GROUP_SIZE = 16
GROUP_BYTES = KEY_SPACE // GROUP_SIZE // 8
PHASE_ZERO = 192  # uint8 phase 3*pi/2: bit 0
PHASE_ONE = 64    # uint8 phase pi/2: bit 1


def key_site(key: int, structured: bool = False) -> int:
    """A PUBLIC query-to-site routing rule. Its cost is shared across queries.

    Structured routing is given privileged knowledge that keys in each block
    of 16 share the same bit. It is not discovered or learned here.
    """
    if not 0 <= key < KEY_SPACE:
        raise ValueError(f"key must be in [0,{KEY_SPACE - 1}]")
    if structured:
        return 1 + key // GROUP_SIZE
    return 1 + ((73 * key + 19) % (PHASE_BYTES - 1))


def encode_bit(bit: int) -> int:
    if bit not in (0, 1):
        raise ValueError("facts are binary")
    return PHASE_ONE if bit else PHASE_ZERO


def uint8_phase(byte: int) -> float:
    return 2 * math.pi * int(byte) / 256


class PhaseGeometry:
    """One 64-byte, 8x8 array of phase delays, with a zero reference port.

    Queries can select a launch site via public key routing, but receive only
    remote measurements at port 0. Collisions overwrite without a key tag.
    """

    def __init__(self, structured: bool = False):
        self.structured = structured
        self.cells = bytearray(PHASE_BYTES)

    def write(self, key: int, bit: int) -> None:
        self.cells[key_site(key, self.structured)] = encode_bit(bit)

    def phase(self, site: int) -> float:
        return uint8_phase(self.cells[site])

    def direct(self, key: int) -> int:
        return int(math.sin(self.phase(key_site(key, self.structured))) > 0)


class BoundedTape:
    """A 64-byte list of 32 two-byte key-value records (not a Transformer)."""

    def __init__(self):
        self.items: list[tuple[int, int]] = []

    def write(self, key: int, bit: int) -> None:
        self.items = [(k, b) for k, b in self.items if k != key]
        self.items.append((key, bit))
        self.items = self.items[-(TAPE_BYTES // TAPE_RECORD_BYTES):]

    def read(self, key: int) -> int | None:
        for k, b in reversed(self.items):
            if k == key:
                return b
        return None


class IndexedBitTable:
    """Exact 256-key bit table: 32 bytes, requiring a preknown key universe."""

    def __init__(self):
        self.bits = bytearray(BITSET_BYTES)

    def write(self, key: int, bit: int) -> None:
        byte, shift = divmod(key, 8)
        mask = 1 << shift
        if bit:
            self.bits[byte] |= mask
        else:
            self.bits[byte] &= ~mask

    def read(self, key: int) -> int:
        byte, shift = divmod(key, 8)
        return (self.bits[byte] >> shift) & 1


class GroupBitTable:
    """2-byte oracle with known 16-key shared-bit groups. Not a learned model."""

    def __init__(self):
        self.bits = bytearray(GROUP_BYTES)

    def write(self, key: int, bit: int) -> None:
        group = key // GROUP_SIZE
        i, shift = divmod(group, 8)
        mask = 1 << shift
        if bit:
            self.bits[i] |= mask
        else:
            self.bits[i] &= ~mask

    def read(self, key: int) -> int:
        group = key // GROUP_SIZE
        i, shift = divmod(group, 8)
        return (self.bits[i] >> shift) & 1


def route_to_port(source: int, steps: int = 12) -> list[int]:
    """Known toroidal nearest-neighbor walk, with stays/loops to fix length.

    This is a query-selected route; choosing the starting site is a strong
    addressing assumption, not an emergent controller. Every intermediate
    location is identical for the two operator orders.
    """
    if not 0 <= source < PHASE_BYTES:
        raise ValueError("source outside grid")
    r, c = divmod(source, GRID_SIDE)
    path = [source]
    while r != 0:
        r = (r - 1 if r <= GRID_SIDE // 2 else r + 1) % GRID_SIDE
        path.append(r * GRID_SIDE + c)
    while c != 0:
        c = (c - 1 if c <= GRID_SIDE // 2 else c + 1) % GRID_SIDE
        path.append(c)
    if len(path) > steps + 1:
        raise ValueError("too few steps to reach port")
    # Fixed-length paths can include identity shifts and short two-step loops.
    while len(path) < steps + 1:
        remaining = steps + 1 - len(path)
        if remaining == 1:
            path.append(PORT)  # identity shift at detector
        else:
            path.extend([1, PORT])  # east and west, physical neighbors
    return path


def ordered_waves(cells: bytes | bytearray, path: list[int], nuisance: list[float] | None = None) -> tuple[complex, complex]:
    """Two unitary responses: phase-then-shift vs shift-then-phase.

    Add an independent nuisance phase to each *intermediate visit*, shared
    across both orders. No perturbation is applied to either endpoint.
    """
    phases = [uint8_phase(cells[p]) for p in path]
    if nuisance is not None:
        if len(nuisance) != len(path) - 2:
            raise ValueError("nuisance must cover interior steps")
        for i, delta in enumerate(nuisance, start=1):
            phases[i] += delta
    return cmath.exp(1j * sum(phases[:-1])), cmath.exp(1j * sum(phases[1:]))


def phase_sign(value: complex) -> int:
    return int(value.imag > 0.0)


def probe_predictions(memory: PhaseGeometry, key: int, rng: random.Random, attack: bool = False) -> dict[str, int]:
    source = key_site(key, memory.structured)
    path = route_to_port(source)
    noise = [rng.uniform(-math.pi, math.pi) for _ in range(len(path) - 2)] if attack else None
    z1, z2 = ordered_waves(memory.cells, path, noise)
    ratio = z1 * z2.conjugate()
    return {
        "direct_addressed_phase": memory.direct(key),
        "single_wave": phase_sign(z1),
        "reversed_wave": phase_sign(z2),
        "order_difference": phase_sign(z1 - z2),
        "paired_ratio": phase_sign(ratio),
        "raw_pair_analytic": phase_sign(z1 * z2.conjugate()),
    }


@dataclass(frozen=True)
class Scenario:
    name: str
    structured: bool  # route uses group sites (public prior)
    correlated: bool | None = None  # whether facts actually obey that prior


def one_trial(seed: int, n: int, scenario: Scenario, *, nuisance: bool = False) -> dict[str, float]:
    """Mean accuracy on ALL written keys (not on unseen keys)."""
    rng = random.Random(seed)
    keys = rng.sample(range(KEY_SPACE), n)
    group_bits = [rng.randrange(2) for _ in range(KEY_SPACE // GROUP_SIZE)]
    obeys_groups = scenario.structured if scenario.correlated is None else scenario.correlated
    facts = {key: group_bits[key // GROUP_SIZE] if obeys_groups else rng.randrange(2) for key in keys}
    memory = PhaseGeometry(structured=scenario.structured)
    tape = BoundedTape()
    bitset = IndexedBitTable()
    group_table = GroupBitTable()
    for key in keys:
        bit = facts[key]
        memory.write(key, bit)
        tape.write(key, bit)
        bitset.write(key, bit)
        if obeys_groups:
            group_table.write(key, bit)
    totals: dict[str, float] = defaultdict(float)
    for key in keys:
        answer = facts[key]
        for name, predicted in probe_predictions(memory, key, rng, attack=nuisance).items():
            totals[name] += float(predicted == answer)
        answer_from_tape = tape.read(key)
        totals["bounded_tape"] += 0.5 if answer_from_tape is None else float(answer_from_tape == answer)
        totals["indexed_bits"] += float(bitset.read(key) == answer)
        totals["full_tape"] += 1.0
        if obeys_groups:
            totals["group_oracle"] += float(group_table.read(key) == answer)
    return {k: v / n for k, v in totals.items()}


def sweep(seed: int, trials: int, counts: Iterable[int], *, nuisance: bool = False) -> list[dict]:
    """Reproducible Monte Carlo receipts, accuracy and sample dispersion."""
    if trials < 1:
        raise ValueError("trials must be positive")
    records = []
    scenarios = (Scenario("independent_bits", False), Scenario("shared_group_bits", True),
                 Scenario("broken_group_rule", True, False))
    for s_idx, scenario in enumerate(scenarios):
        for n in counts:
            if not 1 <= n <= KEY_SPACE:
                raise ValueError(f"fact count must be in 1..{KEY_SPACE}")
            run_results = [one_trial(seed * 1000003 + s_idx * 300011 + n * 1009 + t, n, scenario, nuisance=nuisance) for t in range(trials)]
            names = sorted(run_results[0])
            for name in names:
                values = [v[name] for v in run_results]
                mean = sum(values) / len(values)
                stderr = math.sqrt(sum((x - mean) ** 2 for x in values) / (len(values) - 1) / len(values)) if len(values) > 1 else None
                records.append({"scenario": scenario.name, "facts": n, "reader": name,
                                "accuracy": round(mean, 6), "trial_stderr": None if stderr is None else round(stderr, 6), "trials": trials})
    return records
