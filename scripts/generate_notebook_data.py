"""Generate the small JSON recipe dataset for notebooks/train_sctt.ipynb."""

import argparse
import json
from pathlib import Path
import random


def generate_dataset():
    candidates = [
        [-3, -1.6, 1],
        [-1, -2, 1.1],
        [1, -2, 1],
        [3, -1.6, 1.2],
        [3, 1.6, 1.1],
        [1, 2, 1],
        [-1, 2, 1.2],
        [-3, 1.6, 1],
    ]
    records = []
    for split, size, seed, rounds in [
        ("train", 32, 4100, [0, 1]),
        ("validation", 8, 5200, [2]),
        ("test", 8, 6300, [3]),
    ]:
        rng = random.Random(seed)
        for index in range(size):
            records.append(
                {
                    "id": f"{split}-{index:03d}",
                    "split": split,
                    "round_id": rounds[index % len(rounds)],
                    "feature_seed": seed + index,
                    "valid_features": rng.randint(6, 12),
                    "position_m": [
                        round(rng.uniform(-2.8, 2.8), 5),
                        round(rng.uniform(-1.8, 1.8), 5),
                        round(rng.uniform(0.8, 1.3), 5),
                    ],
                }
            )
    return {
        "schema_version": 1,
        "synthetic": True,
        "description": "Local-metre positions and deterministic observation recipes. Location-encoded synthetic signatures, not camera features.",
        "scale_m": [4.0, 3.0, 2.0],
        "desc_dim": 256,
        "hidden_dim": 64,
        "max_features": 12,
        "top_k": 4,
        "descriptor_projection_seed": 701,
        "candidate_projection_seed": 702,
        "candidates": [{"id": f"anchor-{i + 1:02d}", "position_m": p} for i, p in enumerate(candidates)],
        "records": records,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "sample_data/notebook_training.json",
    )
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(generate_dataset(), indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(generate_dataset()['records'])} synthetic recipes to {args.output.name}")


if __name__ == "__main__":
    main()
