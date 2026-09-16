# Provenance and adaptation scope

Reviewed the `master` branches of `scronfm/SCTTInference`, `scronfm/scttv2_inference`, `scronfm/scttv3`, and `scronfm/scttv3_inference`, plus the related visualization code in `scronfm/scT`.

Reviewed source commits (2026-09-16):

```text
SCTTInference      0558e11b0e23a4c58c263b672923a12dd6b99491
scttv2_inference   9a3ca3c1e824ff3f522a07c4663a75090d84a2b0
scttv3            be1bbfec80fdbbc8ed7c4b9528f88d27a03ea4ca
scttv3_inference  f8f533f2a97db865cfc6038f335eb7c3cb4ff636
scT               856b1d267aeeaccffb1d2be2424c3b29946f0f56
```

| Source | Concepts carried into this standalone implementation |
| --- | --- |
| `SCTTInference/pipes/sctt/` | Descriptor/UV/ray/confidence token projection, candidate-conditioned encoder/decoder and positional refinement |
| `scttv2_inference/sctt_inference/model/` | Ray encoding and compact localization workflow; its `HarmonicEmbedding` dependency is documented rather than copied |
| `scttv3/models/sctt.py` | Candidate embeddings and positions, refinement deltas and relative variance weighting |
| `scttv3_inference` | Inference boundary and position outputs, replacing ROS and private database coupling with JSON/HTTP |
| `scT/sctt/data/` and `scT/sctt/visualization/` | PyTorch3D point-cloud organization, RGB/feature views, filtered candidate overlays and camera poses |

The demo is a fresh, compact adaptation, not a copy of any repository's history or a claim that the production pipelines run unchanged. Legacy entry points have mismatched interfaces and machine-specific dependencies. The source review informed the runnable interface here.

## Data and weights

All positions use an invented local coordinate frame in metres. There are no real map locations, UTM extents, GPS traces, images, identity records, device identifiers or production feature descriptors. The scene, training observations, candidate poses and weights are generated specifically for this demonstration.

The separate `ground_truth` field is used for training/evaluation and display and is not passed into the model. The synthetic descriptors deliberately encode normalized position in their first three channels; this makes the toy task learnable and is not a realistic visual feature pipeline. Synthetic error measurements characterize this toy distribution only. Candidate log-variance values express model-relative weighting and are not validated confidence intervals.

## Excluded material

Original git histories, credentials, database endpoints, private camera calibration, real datasets, production checkpoints, experiment tracking keys, IDE files, machine paths and third-party copied model trees are excluded. Original repositories remain unchanged.

The repository owner has released this showcase, including the adapted notebook, documentation, synthetic demo data and toy-trained weights, under the [MIT License](../LICENSE). This release does not change the licensing of the original repositories. PyTorch, PyTorch3D, Plotly and other dependencies are installed from their own distributions and retain their own notices.

## Supplied training notebook

`notebooks/train_sctt.ipynb` is a reviewed adaptation of the user-supplied notebook with the same basename. Its metadata records the original SHA256 for traceability. The input file remains unchanged. The model class structure, 256-channel descriptor input, residual candidate refinement and masked L1 objective are retained; width/layers and training duration are reduced for CPU execution.

The adaptation replaces private geographic bounds, external storage, archive extraction, experiment tracking and pretrained checkpoint paths with 48 synthetic JSON recipes and local outputs. It fixes CPU-only execution, normalizes candidates and targets consistently, uses separate training/validation/test rounds and labels saved checkpoints as weights-only. Its checkpoint dimensions differ from the web API model. Original execution and environment metadata are not carried over; any saved outputs come from the synthetic adaptation.

## Reference documentation

- [PyTorch3D installation](https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md)
- [PyTorch3D point-cloud visualization and rendering tutorial](https://pytorch3d.org/tutorials/render_colored_points)
- [FastAPI testing](https://fastapi.tiangolo.com/tutorial/testing/)
