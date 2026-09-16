"""Export real PyTorch3D scene components as a self-contained interactive HTML.

    python examples/pytorch3d_demo.py --renderer tensors --output demo.html
    python examples/pytorch3d_demo.py --renderer full --color-mode feature

The tensor path uses genuine Pointclouds + quaternion_to_matrix with a Plotly
adapter. It works with an official no-extension PyTorch3D build. The full path
uses plot_scene + PerspectiveCameras and requires PyTorch3D's compiled library.
See docs/pytorch3d.md for compatible installation instructions.
"""

import argparse
from pathlib import Path

import torch

from sctt_showcase.data import load_scene
from sctt_showcase.model import DemoEngine
from sctt_showcase.visualization import build_figure, build_full_pytorch3d_figure, to_pytorch3d_pointcloud


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--renderer", choices=["tensors", "full"], default="tensors")
    parser.add_argument("--color-mode", choices=["rgb", "feature"], default="rgb")
    parser.add_argument("--sample", default="loop-01")
    parser.add_argument("--top-k", type=int, choices=range(1, 9), default=4)
    parser.add_argument("--output", type=Path, default=Path("pytorch3d_demo.html"))
    parser.add_argument("--open", action="store_true", help="Open the HTML in the default browser")
    args = parser.parse_args()
    torch.set_num_threads(2)
    scene = load_scene()
    result = DemoEngine().infer({"sample_id": args.sample, "top_k": args.top_k})
    try:
        cloud = to_pytorch3d_pointcloud(scene, args.color_mode)
        figure = (
            build_full_pytorch3d_figure(scene, result, color_mode=args.color_mode)
            if args.renderer == "full"
            else build_figure(scene, result, color_mode=args.color_mode, backend="pytorch3d")
        )
    except ImportError as exc:
        parser.exit(2, f"{exc}\n")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.write_html(
        args.output,
        include_plotlyjs=True,
        auto_open=args.open,
        config={"responsive": True, "displaylogo": False},
    )
    print(
        f"PyTorch3D Pointclouds: {tuple(cloud.points_packed().shape)} points, "
        f"{tuple(cloud.features_packed().shape)} RGB features on {cloud.device}"
    )
    print(
        "Renderer: "
        + (
            "PyTorch3D plot_scene + PerspectiveCameras"
            if args.renderer == "full"
            else "PyTorch3D tensors · Plotly view"
        )
    )
    print("Synthetic demonstration only; displayed camera orientations are provided, not predicted.")
    print(f"Saved {args.output.resolve()}")


if __name__ == "__main__":
    main()
