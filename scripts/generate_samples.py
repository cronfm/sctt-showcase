"""Regenerate the small synthetic scene deterministically."""

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sctt_showcase.data import DATA_DIR, build_scene


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    (DATA_DIR / "scene.json").write_text(json.dumps(build_scene(), indent=2) + "\n", encoding="utf-8")
    examples = Path(__file__).resolve().parents[1] / "sample_data"
    examples.mkdir(parents=True, exist_ok=True)
    (examples / "request.json").write_text(
        json.dumps({"sample_id": "loop-01", "top_k": 4, "candidate_offset_m": [0, 0, 0]}, indent=2) + "\n",
        encoding="utf-8",
    )
    scene = build_scene()
    (examples / "custom_request.json").write_text(
        json.dumps(
            {
                "top_k": 4,
                "features": scene["samples"][0]["features"],
                "candidates": [{k: v for k, v in c.items() if k != "score"} for c in scene["candidates"]],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print("Generated package assets/scene.json and sample_data request examples.")


if __name__ == "__main__":
    main()
