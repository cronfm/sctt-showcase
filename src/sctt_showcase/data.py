"""Procedural local-coordinate samples; no source datasets or geolocation."""

from copy import deepcopy
import json
import math
from pathlib import Path
import random

DATA_DIR = Path(__file__).resolve().parent / "assets"
SCALE = [4.0, 3.0, 2.0]
DESC_DIM = 16


def signature(position):
    """Deliberately simple location-encoded signatures, not image descriptors.

    This makes the toy learning problem reproducible without external models.
    The first three channels expose normalized XYZ; this is educational data,
    not evidence of visual localization performance.
    """
    x, y, z = [float(p) / s for p, s in zip(position, SCALE)]
    return [
        x,
        y,
        z,
        math.sin(x),
        math.cos(x),
        math.sin(y),
        math.cos(y),
        math.sin(z),
        math.cos(z),
        x * y,
        y * z,
        z * x,
        x * x,
        y * y,
        z * z,
        1.0,
    ]


def observation(position, seed=0, count=12, yaw=0.0):
    rng = random.Random(seed)
    sig = signature(position)
    desc, uv, rays, scores = [], [], [], []
    for _ in range(count):
        u, v = rng.uniform(-0.8, 0.8), rng.uniform(-0.6, 0.6)
        norm = math.sqrt(u * u + v * v + 1)
        uv.append([round(u, 6), round(v, 6)])
        rays.append(
            [
                round((math.cos(yaw) - u * math.sin(yaw)) / norm, 6),
                round((math.sin(yaw) + u * math.cos(yaw)) / norm, 6),
                round(v / norm, 6),
            ]
        )
        desc.append([round(value + rng.gauss(0, 0.008), 6) for value in sig])
        scores.append(round(rng.uniform(0.7, 1.0), 6))
    return {"desc": desc, "uv": uv, "ray_dir": rays, "score": scores, "mask": [True] * count}


def candidate_records():
    coords = [
        (-3, -1.6, 1),
        (-1, -2, 1.1),
        (1, -2, 1),
        (3, -1.6, 1.2),
        (3, 1.6, 1.1),
        (1, 2, 1),
        (-1, 2, 1.2),
        (-3, 1.6, 1),
    ]
    return [
        {
            "id": f"anchor-{i + 1:02d}",
            "position": list(p),
            "descriptor": [round(v, 6) for v in signature(p)],
            "log_var": [0.0, 0.0, 0.0],
            "quaternion": [0.5, 0.5, 0.5, 0.5],
        }
        for i, p in enumerate(coords)
    ]


def build_scene():
    rng = random.Random(2026)
    xyz, rgb, features = [], [], []
    # A small room: floor, two walls, and two cylindrical landmarks.
    for i in range(160):
        x, y, z = rng.uniform(-4, 4), rng.uniform(-3, 3), 0.0
        xyz.append([round(x, 4), round(y, 4), z])
        rgb.append([65, 94, 118])
        features.append(round((x + 4) / 8, 4))
    for wall in range(2):
        for i in range(70):
            x = rng.uniform(-4, 4) if wall == 0 else -4.0
            y = 3.0 if wall == 0 else rng.uniform(-3, 3)
            z = rng.uniform(0, 2.8)
            xyz.append([round(x, 4), round(y, 4), round(z, 4)])
            rgb.append([74, 158, 173] if wall == 0 else [101, 126, 207])
            features.append(round(z / 2.8, 4))
    for center, color in [((-1.3, 0.6), [249, 180, 83]), ((1.4, 0.2), [206, 99, 137])]:
        for i in range(55):
            theta, z = rng.uniform(0, 2 * math.pi), rng.uniform(0, 2)
            xyz.append(
                [
                    round(center[0] + 0.3 * math.cos(theta), 4),
                    round(center[1] + 0.3 * math.sin(theta), 4),
                    round(z, 4),
                ]
            )
            rgb.append(color)
            features.append(round(z / 2, 4))
    positions = [(-2.2, -1.2, 1.0), (-0.7, -0.9, 1.1), (1.0, -0.7, 1.0), (2.2, 0.5, 1.2)]
    samples, trajectory = [], []
    for i, p in enumerate(positions):
        yaw = (i - 1) * 0.2
        s, c = math.sin(yaw / 2), math.cos(yaw / 2)
        q = [round((c - s) / 2, 6), round((c + s) / 2, 6), round((c + s) / 2, 6), round((c - s) / 2, 6)]
        pose = {"id": f"loop-{i + 1:02d}", "position": list(p), "quaternion": q}
        trajectory.append(pose)
        samples.append(
            {
                "id": pose["id"],
                "label": f"Synthetic pose {i + 1}",
                "position": list(p),
                "ground_truth": list(p),
                "quaternion": q,
                "features": observation(p, seed=100 + i, yaw=yaw),
                "candidate_ids": [c["id"] for c in candidate_records()],
            }
        )
    candidates = candidate_records()
    return {
        "schema_version": 1,
        "name": "Synthetic room / local metres",
        "metadata": {
            "synthetic": True,
            "seed": 2026,
            "units": "metres",
            "coordinate_frame": "local XYZ, world Z up",
            "quaternion_order": "xyzw",
            "camera_axes": "camera +Z forward, +Y up; quaternions map camera to world",
            "descriptor_dim": DESC_DIM,
            "description": "Location-encoded synthetic signatures; not camera-derived descriptors or a real-world benchmark.",
        },
        "points": {"xyz": xyz, "rgb": rgb, "features": features},
        "trajectory": trajectory,
        "query": deepcopy(trajectory[0]),
        "truth": deepcopy(trajectory[0]),
        "candidates": [{**c, "score": 1.0 / len(candidates)} for c in candidates],
        "samples": samples,
        "bounds": [[-4, -3, 0], [4, 3, 3]],
    }


def load_scene():
    with (DATA_DIR / "scene.json").open(encoding="utf-8") as handle:
        return json.load(handle)


def list_samples():
    return [{k: s[k] for k in ("id", "label", "position")} for s in load_scene()["samples"]]


def get_sample(sample_id):
    for sample in load_scene()["samples"]:
        if sample["id"] == sample_id:
            return sample
    raise KeyError(f"Unknown sample: {sample_id}")
