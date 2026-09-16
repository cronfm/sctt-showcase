"""Train only on reproducible toy signatures; downloads no data or weights."""

import argparse
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import torch
from sctt_showcase.data import DATA_DIR, SCALE, candidate_records
from sctt_showcase.model import MODEL_CONFIG, SCTTransformer


def signatures(positions):
    x, y, z = positions.unbind(-1)
    return torch.stack(
        [
            x,
            y,
            z,
            x.sin(),
            x.cos(),
            y.sin(),
            y.cos(),
            z.sin(),
            z.cos(),
            x * y,
            y * z,
            z * x,
            x * x,
            y * y,
            z * z,
            torch.ones_like(x),
        ],
        dim=-1,
    )


def batch(generator, size=48, n=12, k=4):
    target = torch.rand(size, 3, generator=generator)
    target = target * torch.tensor([1.5, 1.4, 0.6]) + torch.tensor([-0.75, -0.7, 0.3])
    sig = signatures(target)
    desc = sig[:, None, :].expand(-1, n, -1) + torch.randn(size, n, 16, generator=generator) * 0.008
    uv = torch.rand(size, n, 2, generator=generator) * 1.2 - 0.6
    rays = torch.nn.functional.normalize(torch.cat([torch.ones(size, n, 1), uv], -1), dim=-1)
    score = torch.rand(size, n, generator=generator) * 0.3 + 0.7
    anchors = torch.tensor([c["position"] for c in candidate_records()]) / torch.tensor(SCALE)
    anchor_desc = signatures(anchors)
    indices = torch.cdist(sig, anchor_desc).topk(k, largest=False).indices
    positions = anchors[indices] + torch.randn(size, k, 3, generator=generator) * 0.055
    feats = {"desc": desc, "uv": uv, "ray_dir": rays, "score": score}
    mask = torch.rand(size, n, generator=generator) > 0.12
    mask[:, 0] = True
    return feats, mask, anchor_desc[indices], positions, torch.zeros(size, k, 3), target


def train(steps=800, seed=2026):
    torch.set_num_threads(2)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    generator = torch.Generator().manual_seed(seed + 1)
    model = SCTTransformer(**MODEL_CONFIG)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.002, weight_decay=0.0001)
    started = time.perf_counter()
    initial = None
    for step in range(steps):
        model.train()
        *inputs, target = batch(generator, k=[1, 2, 4, 8][step % 4])
        result = model(*inputs)
        loss = (result["p0"] - target).square().mean() + 0.00005 * result["log_var"].square().mean()
        if initial is None:
            initial = float(loss.detach())
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if (step + 1) % 200 == 0:
            print(f"step {step + 1}/{steps}: toy normalized loss={loss.item():.6f}", flush=True)
    model.eval()
    holdout = torch.Generator().manual_seed(seed + 100)
    with torch.inference_mode():
        *inputs, target = batch(holdout, size=256, k=4)
        output = model(*inputs)["p0"]
        errors = torch.linalg.vector_norm((output - target) * torch.tensor(SCALE), dim=-1)
    metrics = {
        "scope": "Synthetic signatures with explicit normalized XYZ channels; NOT a real-world localization benchmark.",
        "training_seed": seed,
        "holdout_seed": seed + 100,
        "steps": steps,
        "batch_size": 48,
        "holdout_size": 256,
        "initial_loss": initial,
        "final_loss": float(loss.detach()),
        "holdout_mean_error_m": errors.mean().item(),
        "holdout_p95_error_m": torch.quantile(errors, 0.95).item(),
        "elapsed_seconds": time.perf_counter() - started,
        "torch_version": torch.__version__,
        "model_config": MODEL_CONFIG,
        "uncertainty": "Variance heads are relative weighting signals; not calibrated confidence intervals.",
    }
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(model.cpu().state_dict(), DATA_DIR / "demo_model.pt")
    (DATA_DIR / "training_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=800)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("--steps must be positive")
    train(args.steps, args.seed)
