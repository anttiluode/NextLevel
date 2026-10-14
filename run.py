"""Reproducible offline first gate. Usage: python run.py --trials 50"""
import argparse
import csv
import json
from pathlib import Path
import platform
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from nextlevel.experiment import sweep, KEY_SPACE


def main() -> None:
    parser = argparse.ArgumentParser(description="Capacity and active sensing, not a Transformer benchmark")
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--trials", type=int, default=60)
    parser.add_argument("--facts", type=int, nargs="+", default=[8, 16, 32, 64, 96, 128, 192, 256])
    parser.add_argument("--out", type=Path, default=Path("results/pilot.json"))
    args = parser.parse_args()
    if len(set(args.facts)) != len(args.facts) or any(n < 1 or n > KEY_SPACE for n in args.facts):
        parser.error("fact counts must be unique integers in 1..256")
    rows = sweep(args.seed, args.trials, args.facts)
    attacked = sweep(args.seed, args.trials, args.facts, nuisance=True)
    for row in attacked:
        row["condition"] = "intermediate_phase_perturbed"
    for row in rows:
        row["condition"] = "clean"
    document = {
        "protocol": "gate_0_v1; synthetic fixed known key universe; streaming write; no trained model",
        "seed": args.seed, "trials_per_scenario_count": args.trials, "key_space": KEY_SPACE,
        "budgets_bytes": {"geometry": 64, "bounded_tape": 64, "indexed_bits": 32,
                          "structured_group_oracle": 2, "full_tape_per_fact": 2},
        "queries": "all written keys once per independent trial; missing bounded-tape entry scored 0.5 expected accuracy",
        "important_constraints": [
            "The key-to-site function and routes are public, hard-coded and free/shared; no routing learning is measured.",
            "Structured scenario is GIVEN a 16-key group rule (oracle); it is not learned from examples.",
            "Geometry has 63 data sites + 1 reference port, collisions overwrite, and has no key identity tags.",
            "Each paired measurement uses 2 complex endpoint readings and 2 path traversals; single uses 1.",
            "Indexed bits exploit the small known 256-key domain: 32-byte exact factual memory; this is a strong task-specific baseline.",
            "No learned Transformer, KV cache, neural generalization or wall-clock/energy comparison is present.",
            "Accuracy measures retrieval of facts that were written; not language modeling or unseen world inference."
        ],
        "runtime": {"python": platform.python_version()},
        "results": rows + attacked,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    csv_path = args.out.with_suffix(".csv")
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["scenario", "facts", "reader", "condition", "accuracy", "trial_stderr", "trials"])
        writer.writeheader()
        writer.writerows(rows + attacked)
    print(f"Wrote {args.out} and {csv_path}")
    print("scenario       facts  condition    phase_ratio  single_wave  64B_tape   32B_bitset")
    for scenario in ("independent_bits", "shared_group_bits", "broken_group_rule"):
        for n in args.facts:
            for condition in ("clean", "intermediate_phase_perturbed"):
                subset = {r["reader"]: r["accuracy"] for r in rows + attacked if r["scenario"] == scenario and r["facts"] == n and r["condition"] == condition}
                print(f"{scenario[:14]:14} {n:5}  {condition[:12]:12} {subset['paired_ratio']:10.3f}  {subset['single_wave']:11.3f}  {subset['bounded_tape']:9.3f}   {subset['indexed_bits']:10.3f}")


if __name__ == "__main__":
    main()
