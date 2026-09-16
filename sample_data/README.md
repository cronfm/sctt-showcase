# Minimal request examples

`request.json` runs a bundled sample with four reference candidates.
`custom_request.json` supplies 12 feature rows and eight candidate records directly. Ground truth is not included in the request or passed to the model.

All data is generated in a local coordinate frame measured in metres. Nothing is taken from a vehicle, camera, real location, or source repository dataset. The synthetic 16-channel descriptors deliberately include normalized XYZ in their first three channels. This keeps the learning demonstration small and reproducible; these are not image descriptors or a visual localization benchmark.

The full 410-point scene, four example poses, toy model state dictionary, and training metrics are bundled in `src/sctt_showcase/assets/` for installed-package use. Regenerate with `python scripts/generate_samples.py`, and retrain with `python scripts/train_demo.py`. The model predicts position. Displayed orientation is copied from the sample input.
