import cmath
import math
import random
import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nextlevel.experiment import (
    PhaseGeometry, BoundedTape, IndexedBitTable, GroupBitTable,
    key_site, ordered_waves, probe_predictions, route_to_port, one_trial, Scenario, sweep
)


class FirstGateTests(unittest.TestCase):
    def test_route_cardinal_and_ports(self):
        for source in range(1, 64):
            path = route_to_port(source)
            self.assertEqual((path[0], path[-1], len(path)), (source, 0, 13))
            for a, b in zip(path, path[1:]):
                dr = min(abs(a // 8 - b // 8), 8 - abs(a // 8 - b // 8))
                dc = min(abs(a % 8 - b % 8), 8 - abs(a % 8 - b % 8))
                self.assertIn(dr + dc, (0, 1))

    def test_exact_common_path_invariance(self):
        mem = PhaseGeometry()
        rng = random.Random(9)
        for key in range(75):
            mem.write(key, key % 2)
        for key in range(75):
            path = route_to_port(key_site(key))
            z1, z2 = ordered_waves(mem.cells, path)
            a, b = ordered_waves(mem.cells, path, [rng.uniform(-math.pi, math.pi) for _ in range(11)])
            expected = cmath.exp(1j * (mem.phase(path[0]) - mem.phase(path[-1])))
            self.assertLess(abs(z1 * z2.conjugate() - expected), 1e-10)
            self.assertLess(abs(a * b.conjugate() - expected), 1e-10)

    def test_collisions_and_last_write(self):
        mem = PhaseGeometry()
        self.assertEqual(key_site(1), key_site(64))
        mem.write(1, 0)
        self.assertEqual(mem.direct(1), 0)
        mem.write(64, 1)
        self.assertEqual(mem.direct(64), 1)
        self.assertEqual(mem.direct(1), 1)  # forgotten previous key, not magically retrieved

    def test_exact_bit_table_and_rewrites(self):
        bitset = IndexedBitTable()
        tape = BoundedTape()
        for key in range(256):
            bitset.write(key, key % 2)
            tape.write(key, key % 2)
        self.assertEqual(len(bitset.bits), 32)
        self.assertIsNone(tape.read(0))
        self.assertEqual(tape.read(255), 1)
        for key in range(256):
            self.assertEqual(bitset.read(key), key % 2)
        bitset.write(0, 1)
        tape.write(0, 1)
        self.assertEqual(bitset.read(0), 1)
        self.assertEqual(tape.read(0), 1)

    def test_structured_has_16_sites_and_oracle(self):
        mem = PhaseGeometry(structured=True)
        oracle = GroupBitTable()
        for key in range(256):
            bit = (key // 16) % 2
            mem.write(key, bit)
            oracle.write(key, bit)
        self.assertEqual(len(oracle.bits), 2)
        for key in range(256):
            self.assertEqual(mem.direct(key), (key // 16) % 2)
            self.assertEqual(oracle.read(key), (key // 16) % 2)

    def test_ratio_matches_direct_phase_under_nuisance(self):
        rng = random.Random(13)
        mem = PhaseGeometry()
        for key in range(45):
            mem.write(key, key % 2)
        for key in range(45):
            r = probe_predictions(mem, key, rng, attack=True)
            self.assertEqual(r['paired_ratio'], r['direct_addressed_phase'])
            self.assertEqual(r['raw_pair_analytic'], r['paired_ratio'])

    def test_deterministic_pilot(self):
        a = sweep(23, 3, [8, 32])
        self.assertEqual(a, sweep(23, 3, [8, 32]))
        result = one_trial(15, 256, Scenario('shared_group_bits', True))
        self.assertEqual(result['paired_ratio'], 1)
        self.assertEqual(result['group_oracle'], 1)
        self.assertEqual(result['indexed_bits'], 1)
        result = one_trial(15, 256, Scenario('independent_bits', False))
        self.assertEqual(result['indexed_bits'], 1)
        self.assertLess(result['paired_ratio'], 0.8)


if __name__ == '__main__':
    unittest.main()
