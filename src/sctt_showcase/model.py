"""Compact adaptation of SCTT's candidate-conditioned encoder/decoder.

The bundled weights learn synthetic location-encoded signatures only. They are
not production localization weights and output variance is not calibrated.
"""

import json
import math
import time

import torch
from torch import nn

from .data import DATA_DIR, SCALE, get_sample, load_scene
from .schemas import InferRequest

MODEL_CONFIG = {"desc_dim": 16, "hidden_dim": 32, "nhead": 4, "num_layers": 1}


class SCTTransformer(nn.Module):
    def __init__(self, desc_dim=16, hidden_dim=32, nhead=4, num_layers=1):
        super().__init__()
        self.feature_projector = nn.Sequential(
            nn.Linear(desc_dim + 6, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim)
        )
        self.candidate_projector = nn.Linear(desc_dim, hidden_dim)
        self.candidate_pos_enc = nn.Linear(3, hidden_dim)
        pe = torch.zeros(128, hidden_dim)
        position = torch.arange(128).unsqueeze(1)
        div = torch.exp(torch.arange(0, hidden_dim, 2) * (-math.log(10000.0) / hidden_dim))
        pe[:, 0::2] = torch.sin(position * div)
        pe[:, 1::2] = torch.cos(position * div)
        self.register_buffer("positional_encoding", pe.unsqueeze(0))
        encoder = nn.TransformerEncoderLayer(
            hidden_dim, nhead, hidden_dim * 2, dropout=0, batch_first=True, activation="gelu"
        )
        decoder = nn.TransformerDecoderLayer(
            hidden_dim, nhead, hidden_dim * 2, dropout=0, batch_first=True, activation="gelu"
        )
        self.encoder = nn.TransformerEncoder(encoder, num_layers, enable_nested_tensor=False)
        self.decoder = nn.TransformerDecoder(decoder, num_layers)
        self.position_refiner = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2), nn.GELU(), nn.Linear(hidden_dim * 2, 6)
        )

    def forward(self, feats, mask, candidate_embeddings, candidate_positions, candidate_log_vars):
        if not mask.any(dim=1).all():
            raise ValueError("Every batch item must contain a valid feature")
        x = torch.cat([feats["desc"], feats["uv"], feats["ray_dir"], feats["score"].unsqueeze(-1)], dim=-1)
        src = self.feature_projector(x) + self.positional_encoding[:, : x.shape[1]]
        memory = self.encoder(src, src_key_padding_mask=~mask.bool())
        target = self.candidate_projector(candidate_embeddings) + self.candidate_pos_enc(candidate_positions)
        decoded = self.decoder(target, memory, memory_key_padding_mask=~mask.bool())
        updates = self.position_refiner(decoded)
        refined = candidate_positions + updates[..., :3]
        log_vars = (candidate_log_vars + updates[..., 3:]).clamp(-8, 8)
        weights = torch.softmax(-log_vars, dim=1)
        return {
            "p0": (refined * weights).sum(dim=1),
            "log_var": (log_vars * weights).sum(dim=1),
            "candidate_weights": weights,
            "candidate_refined_positions": refined,
        }


class DemoEngine:
    def __init__(self, device="cpu"):
        self.device = torch.device(device)
        self.model = SCTTransformer(**MODEL_CONFIG).to(self.device)
        checkpoint = DATA_DIR / "demo_model.pt"
        self.model.load_state_dict(torch.load(checkpoint, map_location=self.device, weights_only=True))
        self.model.eval()
        with (DATA_DIR / "training_metrics.json").open(encoding="utf-8") as handle:
            self.metrics = json.load(handle)
        self.info = {
            "status": "synthetic-trained",
            "checkpoint": checkpoint.name,
            **MODEL_CONFIG,
            "device": str(self.device),
            "scope": "Toy location-encoded signatures only; no real-world localization claim.",
        }

    @torch.inference_mode()
    def infer(self, request: InferRequest | dict):
        if isinstance(request, dict):
            request = InferRequest.model_validate(request)
        started = time.perf_counter()
        if request.features is None:
            sample = get_sample(request.sample_id)
            features = sample["features"]
            all_candidates = load_scene()["candidates"]
            truth, quaternion = sample["ground_truth"], sample["quaternion"]
        else:
            features = request.features.model_dump()
            all_candidates = [c.model_dump() for c in request.candidates]
            truth, quaternion = None, [0.0, 0.0, 0.0, 1.0]
        mask = features.get("mask") or [True] * len(features["desc"])
        valid = [row for row, keep in zip(features["desc"], mask) if keep]
        mean = [sum(row[d] for row in valid) / len(valid) for d in range(16)]
        candidates = sorted(
            all_candidates, key=lambda c: sum((a - b) ** 2 for a, b in zip(mean, c["descriptor"]))
        )[: request.top_k]
        positions = [
            [v + off for v, off in zip(c["position"], request.candidate_offset_m)] for c in candidates
        ]

        def tensor(value):
            return torch.tensor(value, dtype=torch.float32, device=self.device).unsqueeze(0)

        scale = torch.tensor(SCALE, dtype=torch.float32, device=self.device)
        output = self.model(
            {k: tensor(features[k]) for k in ("desc", "uv", "ray_dir", "score")},
            torch.tensor([mask], dtype=torch.bool, device=self.device),
            tensor([c["descriptor"] for c in candidates]),
            tensor(positions) / scale,
            tensor([c["log_var"] for c in candidates]),
        )
        position = (output["p0"][0] * scale).cpu().tolist()
        weights = output["candidate_weights"][0].cpu().tolist()
        refined = (output["candidate_refined_positions"][0] * scale).cpu().tolist()
        error = math.dist(position, truth) if truth is not None else None
        return {
            "sample_id": request.sample_id if request.features is None else "custom",
            "position": position,
            "ground_truth": truth,
            "error_m": error,
            "prediction": {
                "position": position,
                "quaternion": quaternion,
                "orientation_source": (
                    "copied sample pose; not predicted"
                    if request.features is None
                    else "identity display placeholder; not predicted"
                ),
            },
            "truth": {"position": truth, "quaternion": quaternion} if truth is not None else None,
            "candidate_ids": [c["id"] for c in candidates],
            "candidate_positions": positions,
            "candidate_weights": weights,
            "candidate_refined_positions": refined,
            "candidates": [
                {"id": c["id"], "position": p, "quaternion": c["quaternion"], "score": sum(w) / 3}
                for c, p, w in zip(candidates, positions, weights)
            ],
            "model": self.info,
            "uncertainty": {
                "kind": "relative candidate weighting; not calibrated uncertainty",
                "log_variance": output["log_var"][0].cpu().tolist(),
            },
            "duration_ms": (time.perf_counter() - started) * 1000,
        }
