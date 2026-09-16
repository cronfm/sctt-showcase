# Minimal request examples

`request.json` runs a bundled sample with four reference candidates.
`custom_request.json` supplies 12 feature rows and eight candidate records directly. Ground truth is not included in the request or passed to the model.

All data is generated in a local coordinate frame measured in metres. Nothing is taken from a vehicle, camera, real location, or source repository dataset. The synthetic 16-channel descriptors deliberately include normalized XYZ in their first three channels. This keeps the learning demonstration small and reproducible; these are not image descriptors or a visual localization benchmark.

The full 410-point scene, four example poses, toy model state dictionary, and training metrics are bundled in `src/sctt_showcase/assets/` for installed-package use. Regenerate with `python scripts/generate_samples.py`, and retrain with `python scripts/train_demo.py`. The model predicts position. Displayed orientation is copied from the sample input.

## Training notebook data

`notebook_training.json` is a separate 12 KB recipe dataset for `notebooks/train_sctt.ipynb`: 32 training, 8 validation and 8 test positions in disjoint synthetic rounds. It includes candidate anchors, per-observation seeds and valid feature counts. The notebook deterministically expands these recipes into 256-channel descriptors and 64-channel candidate embeddings while padding feature sequences to 12 rows. They remain position-encoded synthetic signatures, not real camera observations.

Regenerate this file with `python scripts/generate_notebook_data.py`. It is training input for the supplied notebook adaptation, not a request body for the HTTP API.
