# PyTorch3D demonstration

The optional backend uses real `pytorch3d.structures.Pointclouds` for the XYZ/RGB batch and real `pytorch3d.transforms.quaternion_to_matrix` for camera transforms. Our Plotly adapter displays those tensors in an interactive browser view. PyTorch3D uses `w,x,y,z` quaternions; the JSON API uses `x,y,z,w`, with explicit conversion at the boundary.

This path does **not** claim to use PyTorch3D's native rasterizer. The browser draws Plotly `scatter3d` traces. You can run it on a CPU without compiling PyTorch3D's native extension.

## Lightweight, tested installation

Install the main package first using the README, then:

```bash
python -m pip install setuptools wheel packaging iopath==0.1.10
```

PowerShell:

```powershell
$env:PYTORCH3D_NO_EXTENSION = '1'
python -m pip install --no-build-isolation 'https://github.com/facebookresearch/pytorch3d/archive/33824be3cbc87a7dd1db0f6a9a9de9ac81b2d0ba.zip#sha256=9f53ec2bb1ab11b3e0d8480ba5f4361407b5649cae26036fefaefb7f301e3c56'
Remove-Item Env:PYTORCH3D_NO_EXTENSION
```

Linux/macOS:

```bash
PYTORCH3D_NO_EXTENSION=1 python -m pip install --no-build-isolation 'https://github.com/facebookresearch/pytorch3d/archive/33824be3cbc87a7dd1db0f6a9a9de9ac81b2d0ba.zip#sha256=9f53ec2bb1ab11b3e0d8480ba5f4361407b5649cae26036fefaefb7f301e3c56'
```

Restart the API and select the PyTorch3D backend in the viewer. `GET /api/capabilities` reports availability. `POST /api/figure` accepts `"backend":"pytorch3d"`. Without the dependency, that request returns 503 with a setup hint; it never silently falls back.

Verified locally with Python 3.12.14, PyTorch 2.14.0+cpu, PyTorch3D 0.7.9, NumPy 2.4.3 and Plotly 6.6.0 on Windows. Source is pinned to the official v0.7.9 commit and archive checksum. The requirements listed in the main package allow other versions; they are not all tested combinations.

## Full PyTorch3D renderer

The optional `examples/pytorch3d_demo.py` demonstrates the lightweight adapter and a separate full-library path. Run `python examples/pytorch3d_demo.py --help` for exact options.

PyTorch3D's `plot_scene`, `renderer.HarmonicEmbedding` and native rasterization import the compiled extension. They are unavailable in the no-extension installation above. To use the full path, follow the [official installation instructions](https://github.com/facebookresearch/pytorch3d/blob/33824be3cbc87a7dd1db0f6a9a9de9ac81b2d0ba/INSTALL.md), select compatible PyTorch/torchvision versions, install an appropriate compiler, and rebuild without `PYTORCH3D_NO_EXTENSION=1`.

The full native build was attempted on the development machine and failed because Microsoft Visual C++ build tools were absent. The compiled renderer example is provided for a configured environment; it was not executed successfully here. The tested showcase uses the real point-cloud/transform API with the Plotly adapter.

Reference: [official point-cloud tutorial](https://pytorch3d.org/tutorials/render_colored_points).
