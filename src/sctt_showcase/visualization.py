"""Interactive scene views with a portable Plotly and a genuine PyTorch3D path.

All poses are camera-to-world: position in metres, quaternion in xyzw order,
camera +Z forward and +Y up. No renderer, CUDA runtime or external service is
required for the default backend. PyTorch3D is imported only when requested.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import plotly.graph_objects as go
from plotly.colors import sample_colorscale

Pose = Mapping[str, Any]
COLORS = {"trajectory": "#708799", "candidate": "#c49b5b", "prediction": "#ed6d42", "truth": "#149482"}


def quaternion_matrix(quaternion: Sequence[float]) -> list[list[float]]:
    """Return the rotation matrix for an xyzw quaternion, normalizing its norm."""
    if len(quaternion) != 4 or not all(math.isfinite(v) for v in quaternion):
        raise ValueError("A quaternion must contain four finite xyzw values")
    norm = math.sqrt(sum(value * value for value in quaternion))
    if norm < 1e-12:
        raise ValueError("A quaternion must have a nonzero norm")
    x, y, z, w = (value / norm for value in quaternion)
    return [
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ]


def pytorch3d_quaternion_matrix(quaternion: Sequence[float]) -> list[list[float]]:
    """Use PyTorch3D's real transform API after converting xyzw to wxyz."""
    quaternion_matrix(quaternion)  # Validate before normalizing the tensor.
    try:
        import torch
        from pytorch3d.transforms import quaternion_to_matrix
    except (ImportError, OSError) as exc:
        raise ImportError("PyTorch3D is optional. See docs/pytorch3d.md or select Plotly.") from exc
    x, y, z, w = quaternion
    tensor = torch.tensor([w, x, y, z], dtype=torch.float32)
    return quaternion_to_matrix(tensor / tensor.norm()).tolist()


def camera_frustum(pose: Pose, scale: float = 0.22, *, backend: str = "plotly") -> list[list[float] | None]:
    """Transform a small camera wireframe into world space; None separates edges."""
    quaternion = pose.get("quaternion", [0, 0, 0, 1])
    rotation = (
        pytorch3d_quaternion_matrix(quaternion) if backend == "pytorch3d" else quaternion_matrix(quaternion)
    )
    position = pose["position"]
    vertices = [(0, 0, 0), (-0.7, -0.45, 1), (0.7, -0.45, 1), (0.7, 0.45, 1), (-0.7, 0.45, 1)]
    world = [
        [position[axis] + scale * sum(rotation[axis][j] * point[j] for j in range(3)) for axis in range(3)]
        for point in vertices
    ]
    edges = [(0, 1), (0, 2), (0, 3), (0, 4), (1, 2), (2, 3), (3, 4), (4, 1)]
    return [point for a, b in edges for point in (world[a], world[b], None)]


def _xyz(points: Sequence[Sequence[float] | None]) -> dict[str, list]:
    return {
        axis: [point[i] if point is not None else None for point in points] for i, axis in enumerate("xyz")
    }


def _pose_trace(
    poses: Sequence[Pose],
    name: str,
    color: str,
    *,
    width: int = 3,
    scale: float = 0.22,
    backend: str = "plotly",
) -> go.Scatter3d:
    points = [point for pose in poses for point in camera_frustum(pose, scale, backend=backend)]
    return go.Scatter3d(
        **_xyz(points),
        mode="lines",
        line={"color": color, "width": width},
        name=name,
        hoverinfo="skip",
        legendgroup=name,
    )


def _feature_colors(values: Sequence[float]) -> list[str]:
    if not values:
        return []
    low, high = min(values), max(values)
    normalized = [(value - low) / (high - low) if high > low else 0.5 for value in values]
    return sample_colorscale("Viridis", normalized)


def to_pytorch3d_pointcloud(scene: Mapping[str, Any], color_mode: str = "rgb"):
    """Build a CPU ``pytorch3d.structures.Pointclouds`` with RGB features.

    Scalar feature values are mapped through Viridis because ``plot_scene``
    consumes RGB/RGBA Pointclouds features. The original scalar is retained in
    the scene, and the portable view exposes it on hover.
    """
    if color_mode not in {"rgb", "feature"}:
        raise ValueError("color_mode must be 'rgb' or 'feature'")
    try:
        import torch
        from pytorch3d.structures import Pointclouds
    except (ImportError, OSError) as exc:
        raise ImportError(
            "PyTorch3D is optional. Install a compatible PyTorch3D build using docs/pytorch3d.md, or choose the Plotly backend."
        ) from exc
    points = scene["points"]
    if color_mode == "rgb":
        colors = [[channel / 255.0 for channel in rgb] for rgb in points["rgb"]]
    else:
        colors = [
            [float(channel) / 255.0 for channel in rgb.removeprefix("rgb(").removesuffix(")").split(",")]
            for rgb in _feature_colors(points["features"])
        ]
    return Pointclouds(
        points=[torch.tensor(points["xyz"], dtype=torch.float32)],
        features=[torch.tensor(colors, dtype=torch.float32)],
    )


def build_full_pytorch3d_figure(
    scene: Mapping[str, Any], result: Mapping[str, Any] | None = None, *, color_mode: str = "rgb"
) -> go.Figure:
    """Full-library path using actual ``plot_scene`` and ``PerspectiveCameras``.

    Unlike ``build_figure(backend='pytorch3d')``, importing this renderer requires
    a PyTorch3D installation with its compiled extension. No extension is faked.
    """
    cloud = to_pytorch3d_pointcloud(scene, color_mode)
    try:
        import torch
        from pytorch3d.renderer import PerspectiveCameras
        from pytorch3d.vis.plotly_vis import plot_scene
    except (ImportError, OSError) as exc:
        raise ImportError(
            "The installed PyTorch3D renderer is unavailable. See docs/pytorch3d.md or select Plotly."
        ) from exc
    structures = {"RGB point cloud" if color_mode == "rgb" else "Feature point cloud": cloud}
    poses = scene.get("trajectory", [])
    if poses:
        # PyTorch3D transforms row vectors as X_camera = X_world @ R + T.
        # Our column-vector camera-to-world matrix is therefore its R directly.
        rotation = torch.tensor(
            [quaternion_matrix(pose["quaternion"]) for pose in poses], dtype=torch.float32
        )
        centers = torch.tensor([pose["position"] for pose in poses], dtype=torch.float32)
        translation = -torch.bmm(centers[:, None, :], rotation)[:, 0, :]
        structures["Trajectory cameras"] = PerspectiveCameras(R=rotation, T=translation, device="cpu")
    figure = plot_scene(
        {"Synthetic room": structures},
        pointcloud_max_points=len(scene["points"]["xyz"]),
        pointcloud_marker_size=3,
        camera_scale=0.09,
    )
    adapter = build_figure(scene, result, color_mode=color_mode, backend="pytorch3d")
    for trace in adapter.data:
        if trace.name not in {"RGB point cloud", "Feature point cloud", "Trajectory cameras"}:
            figure.add_trace(trace)
    figure.update_layout(adapter.layout)
    figure.update_layout(
        meta={"backend": "pytorch3d-plot_scene", "synthetic": True, "quaternion_order": "xyzw"}
    )
    return figure


def build_figure(
    scene: Mapping[str, Any],
    result: Mapping[str, Any] | None = None,
    *,
    color_mode: str = "rgb",
    backend: str = "plotly",
) -> go.Figure:
    """Draw point features, retrieved poses, trajectory and inference overlays.

    ``result`` is the response from ``DemoEngine.infer``. Without a result the
    scene's candidates/truth are shown. Both backends return a Plotly Figure;
    the PyTorch3D branch uses actual Pointclouds packed tensors and quaternion
    transforms. Its adapter also works with an official no-extension build.
    ``build_full_pytorch3d_figure`` demonstrates the separate full-library path.
    """
    if color_mode not in {"rgb", "feature"}:
        raise ValueError("color_mode must be 'rgb' or 'feature'")
    if backend not in {"plotly", "pytorch3d"}:
        raise ValueError("backend must be 'plotly' or 'pytorch3d'")
    points = scene["points"]
    if (
        not points["xyz"]
        or len(points["xyz"]) != len(points["rgb"])
        or len(points["xyz"]) != len(points["features"])
    ):
        raise ValueError("Scene requires matching nonempty xyz, rgb and features arrays")
    marker: dict[str, Any] = {"size": 3, "opacity": 0.86}
    xyz = points["xyz"]
    if backend == "pytorch3d":
        cloud = to_pytorch3d_pointcloud(scene, color_mode)
        xyz = cloud.points_packed().cpu().tolist()
        colors = cloud.features_packed().clamp(0, 1).mul(255).round().cpu().tolist()
        marker["color"] = [f"rgb({int(r)},{int(g)},{int(b)})" for r, g, b in colors]
    elif color_mode == "rgb":
        marker["color"] = [f"rgb({r},{g},{b})" for r, g, b in points["rgb"]]
    else:
        marker.update(
            color=points["features"],
            colorscale="Viridis",
            showscale=True,
            colorbar={"title": "Feature", "thickness": 10, "len": 0.5, "x": 0.96},
        )
    figure = go.Figure(
        go.Scatter3d(
            **_xyz(xyz),
            mode="markers",
            marker=marker,
            customdata=points["features"],
            name="RGB point cloud" if color_mode == "rgb" else "Feature point cloud",
            hovertemplate="x %{x:.2f} · y %{y:.2f} · z %{z:.2f} m<br>feature %{customdata:.3f}<extra>synthetic point</extra>",
        )
    )
    if backend == "pytorch3d" and color_mode == "feature":
        figure.add_trace(
            go.Scatter3d(
                x=[None, None],
                y=[None, None],
                z=[None, None],
                mode="markers",
                marker={
                    "color": [min(points["features"]), max(points["features"])],
                    "colorscale": "Viridis",
                    "showscale": True,
                    "colorbar": {"title": "Feature", "thickness": 10, "len": 0.5, "x": 0.96},
                },
                hoverinfo="skip",
                name="Feature scale",
                showlegend=False,
            )
        )
    if scene.get("trajectory"):
        figure.add_trace(
            _pose_trace(
                scene["trajectory"],
                "Trajectory cameras",
                COLORS["trajectory"],
                width=2,
                scale=0.12,
                backend=backend,
            )
        )
    trajectory = scene.get("trajectory", [])
    if trajectory:
        figure.add_trace(
            go.Scatter3d(
                **_xyz([pose["position"] for pose in trajectory]),
                mode="lines",
                line={"color": COLORS["trajectory"], "width": 3, "dash": "dot"},
                name="Reference trajectory",
                hoverinfo="skip",
            )
        )
    overlays = result if result is not None else scene
    candidates = overlays.get("candidates", [])
    if not candidates and overlays.get("candidate_positions"):
        candidates = [
            {"position": position, "quaternion": [0, 0, 0, 1], "id": index}
            for index, position in enumerate(overlays["candidate_positions"])
        ]
    if candidates:
        figure.add_trace(
            _pose_trace(candidates, "Retrieved cameras", COLORS["candidate"], width=4, backend=backend)
        )
        weights = overlays.get("candidate_weights", [])
        text = []
        for index, candidate in enumerate(candidates):
            label = f"Candidate {candidate.get('id', index + 1)}"
            if index < len(weights):
                label += "<br>axis weights " + ", ".join(f"{value:.3f}" for value in weights[index])
            text.append(label)
        figure.add_trace(
            go.Scatter3d(
                **_xyz([pose["position"] for pose in candidates]),
                mode="markers",
                marker={"color": COLORS["candidate"], "size": 5},
                text=text,
                hovertemplate="%{text}<extra></extra>",
                name="Top-K candidates",
                showlegend=False,
            )
        )
    refined = overlays.get("candidate_refined_positions", [])
    if refined and candidates:
        paths = [
            p
            for candidate, refined_position in zip(candidates, refined)
            for p in (candidate["position"], refined_position, None)
        ]
        figure.add_trace(
            go.Scatter3d(
                **_xyz(paths),
                mode="lines",
                line={"color": COLORS["prediction"], "width": 2, "dash": "dash"},
                name="Learned refinement",
                hoverinfo="skip",
            )
        )
    prediction = overlays.get("prediction")
    if prediction is None and overlays.get("position") is not None:
        prediction = {"position": overlays["position"], "quaternion": [0, 0, 0, 1]}
    truth = overlays.get("truth")
    if truth is None and overlays.get("ground_truth") is not None:
        truth = {"position": overlays["ground_truth"], "quaternion": [0, 0, 0, 1]}
    for pose, label, color, symbol in [
        (prediction, "Predicted position", COLORS["prediction"], "diamond"),
        (truth, "Synthetic ground truth", COLORS["truth"], "circle"),
    ]:
        if pose:
            if "placeholder" not in pose.get("orientation_source", ""):
                figure.add_trace(
                    _pose_trace(
                        [pose], f"{label} orientation (provided)", color, width=5, scale=0.3, backend=backend
                    )
                )
            figure.add_trace(
                go.Scatter3d(
                    **_xyz([pose["position"]]),
                    mode="markers",
                    marker={"color": color, "size": 7, "symbol": symbol},
                    name=label,
                    hovertemplate=label + "<br>(%{x:.3f}, %{y:.3f}, %{z:.3f}) m<extra></extra>",
                )
            )
    if prediction and truth:
        figure.add_trace(
            go.Scatter3d(
                **_xyz([prediction["position"], truth["position"]]),
                mode="lines",
                line={"color": COLORS["truth"], "width": 4},
                name="Position error",
                hoverinfo="skip",
            )
        )
    axes = {
        "backgroundcolor": "#f7faf9",
        "gridcolor": "#dae3e5",
        "zerolinecolor": "#c4d0d4",
        "showbackground": True,
        "color": "#667780",
    }
    figure.update_layout(
        title=None,
        annotations=[],
        paper_bgcolor="#f7faf9",
        plot_bgcolor="#f7faf9",
        font={"family": "Arial, sans-serif", "color": "#334c59", "size": 11},
        margin={"l": 0, "r": 0, "t": 6, "b": 0},
        height=580,
        scene={
            "xaxis": {**axes, "title": "X · m", "autorange": True},
            "yaxis": {**axes, "title": "Y · m", "autorange": True},
            "zaxis": {**axes, "title": "Z · m", "autorange": True},
            "aspectmode": "data",
            "bgcolor": "#f7faf9",
            "camera": {"eye": {"x": 1.5, "y": -1.75, "z": 1.1}, "up": {"x": 0, "y": 0, "z": 1}},
        },
        legend={"orientation": "h", "yanchor": "top", "y": -0.02, "x": 0.02, "font": {"size": 10}},
        uirevision="synthetic-room",
        meta={"backend": backend, "synthetic": True, "quaternion_order": "xyzw"},
    )
    return figure
