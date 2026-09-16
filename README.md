# SCTT Showcase

A small, self-contained demonstration of candidate-conditioned camera localization in PyTorch. Explore a synthetic 3D scene, inspect a transformer's candidate refinements, and call the same model through a REST API.

**Synthetic data and synthetic-trained weights only.** This illustrates the SCTT architecture and visualization workflow; it is not a production localizer or an accuracy benchmark. The repository starts private. Publishing it later is a separate decision.

## Run locally

Python 3.11 or newer. A CPU is enough; no database, ROS installation, camera, credentials, or external model downloads are needed at runtime.

```bash
python -m venv .venv
# Linux / macOS:
source .venv/bin/activate
# Windows PowerShell: .\.venv\Scripts\Activate.ps1
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e ".[dev]"
sctt-showcase
```

Open **http://127.0.0.1:8000** for the interactive viewer and **http://127.0.0.1:8000/docs** for the API explorer. The viewer serves its Plotly JavaScript locally. Swagger's API explorer may load its UI assets from a CDN.

## What to explore

- RGB and feature-colored synthetic point clouds.
- Camera trajectory, orientation frames and viewing frusta in a local metre coordinate system.
- Candidate positions, transformer refinements, the final estimate, and synthetic ground truth.
- Different samples, candidate counts and candidate offsets.
- A genuine optional **PyTorch3D `Pointclouds` and quaternion-transform path**, displayed through Plotly. See [PyTorch3D setup](docs/pytorch3d.md) for the lightweight path and the separate compiled renderer example.

The core viewer works without PyTorch3D. Its backend selector reports whether the optional package is installed; it does not silently substitute another implementation when PyTorch3D is requested.

## Call the API

```bash
curl http://127.0.0.1:8000/api/health
curl http://127.0.0.1:8000/api/samples
curl -X POST http://127.0.0.1:8000/api/infer \
  -H "Content-Type: application/json" \
  -d '{"sample_id":"loop-01","top_k":4,"candidate_offset_m":[0,0,0]}'
```

PowerShell:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/api/infer -Method Post `
  -ContentType 'application/json' `
  -Body '{"sample_id":"loop-01","top_k":4,"candidate_offset_m":[0,0,0]}'
```

| Route | Purpose |
| --- | --- |
| `GET /api/health` | Service version, CPU execution and synthetic-data status |
| `GET /api/capabilities` | Optional PyTorch3D availability |
| `GET /api/scene` | Small synthetic point cloud, trajectory and candidates |
| `GET /api/samples` | Available synthetic observation samples |
| `GET /api/example` | Minimal inference request |
| `POST /api/infer` | Candidate-conditioned position estimate and diagnostics |
| `POST /api/figure` | Plotly figure JSON, using `color_mode` and `backend` |

The generated OpenAPI schema at `/openapi.json` describes custom feature/candidate requests. Requests are bounded to 256 KiB and tensor fields are validated before inference. Custom observations and candidates must be supplied together; their outputs are not scored against a canned sample's ground truth.

## Architecture and limits

The original SCTT family encodes image descriptors, UV coordinates, world rays and feature confidence, then uses candidate-conditioned attention to refine a camera position. This showcase retains those ideas with a compact CPU transformer, 16-dimensional synthetic descriptors and a small candidate set.

The bundled model is trained only on generated local geometry. Its synthetic descriptors deliberately include normalized XYZ in their first three channels, making this a small learnable demonstration rather than an image-localization benchmark. It does not include SuperPoint, LightGlue, segmentation, real camera imagery, GPS, private training data, production checkpoints or a database. Relative candidate weighting is **not calibrated uncertainty**. Changing candidates beyond the training distribution may produce poor estimates; the viewer is designed to make that behavior inspectable.

See [provenance and scope](docs/provenance.md) and the generated training metadata alongside the assets for details. Real-world localization requires a separately trained model and matching feature/calibration pipeline.

## Train in the notebook

[notebooks/train_sctt.ipynb](notebooks/train_sctt.ipynb) adapts the supplied training notebook for a short CPU run. It keeps the original 256-channel feature projector, candidate-conditioned encoder/decoder, refinement, masked L1 loss and warmup/cosine/tail schedule, with a smaller 64-unit, one-layer model.

```bash
python -m pip install -e ".[notebook]"
jupyter lab notebooks/train_sctt.ipynb
```

Restart the kernel and run all cells. The included [demo dataset](sample_data/notebook_training.json) has 48 deterministic observation recipes: 32 training, 8 validation and 8 test examples in disjoint synthetic rounds. Position-encoded signatures are expanded to 256 dimensions; these are not real image descriptors. The notebook trains from scratch, plots learning curves, evaluates held-out examples and visualizes its own predictions in 3D.

Outputs go to ignored `generated/notebook_demo/`: best/last tensor-only weights, metrics and a self-contained interactive scene. The notebook checkpoint uses different dimensions from the API model and does not replace it. Cloud mounts, tracking services and private inputs are removed. To verify the complete notebook in a fresh kernel:

```bash
python scripts/check_notebook.py
```

Regenerate its small input dataset with `python scripts/generate_notebook_data.py`.

## Development

```bash
python -m pytest
python -m ruff check .
python -m build
```

Sample generation and toy training are reproducible through `scripts/generate_samples.py` and `scripts/train_demo.py`; use `--help` for arguments. Data and checkpoint assets are included in the wheel so the installed package also works outside the checkout.

This is a local demonstration service. It binds to `127.0.0.1` by default. Hosting an internet-accessible API requires deployment controls such as authentication and rate limiting; no service is deployed by creating this repository.

## Repository contents

```text
src/sctt_showcase/    API, model, schemas, synthetic data loader, visualization
static/              Interactive browser viewer
sample_data/         Minimal example input
examples/            Optional PyTorch3D demonstration
notebooks/           CPU training notebook adapted from the supplied original
scripts/             Synthetic-data and toy-training reproduction
tests/               Inference, validation, API and visualization checks
docs/                Provenance, PyTorch3D setup and release notes
```

No open-source license has been selected for this showcase. Dependency licenses remain with their respective authors; third-party model code and weights have not been vendored.
