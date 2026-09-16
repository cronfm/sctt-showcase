"""Geometry and optional-library integration tests, without a GPU or browser."""

import json
import math

import pytest

from sctt_showcase.data import load_scene
from sctt_showcase.visualization import (
    build_figure,
    build_full_pytorch3d_figure,
    camera_frustum,
    pytorch3d_quaternion_matrix,
    quaternion_matrix,
    to_pytorch3d_pointcloud,
)


def test_xyzw_rotation_and_translation():
    # A 90-degree rotation about Z maps camera X to world Y.
    q = [0, 0, math.sqrt(0.5), math.sqrt(0.5)]
    matrix = quaternion_matrix(q)
    assert matrix[0] == pytest.approx([0, -1, 0])
    assert matrix[1] == pytest.approx([1, 0, 0])
    points = camera_frustum({"position": [1, 2, 3], "quaternion": q}, scale=1)
    assert points[0] == [1, 2, 3]
    assert points[1] == pytest.approx([1.45, 1.3, 4])
    assert points[2] is None
    with pytest.raises(ValueError, match="nonzero"):
        quaternion_matrix([0, 0, 0, 0])


@pytest.mark.parametrize("color_mode", ["rgb", "feature"])
def test_plotly_scene_includes_geometry_and_truth(color_mode):
    scene = load_scene()
    figure = build_figure(scene, color_mode=color_mode)
    document = json.loads(figure.to_json())
    names = [trace["name"] for trace in document["data"]]
    assert "Trajectory cameras" in names
    assert "Retrieved cameras" in names
    assert "Synthetic ground truth" in names
    assert len(document["data"][0]["x"]) == len(scene["points"]["xyz"])
    assert document["layout"]["meta"]["synthetic"] is True


def test_custom_input_never_inherits_example_truth_or_placeholder_orientation():
    result = {
        "position": [1, 2, 1],
        "ground_truth": None,
        "truth": None,
        "prediction": {
            "position": [1, 2, 1],
            "quaternion": [0, 0, 0, 1],
            "orientation_source": "identity display placeholder; not predicted",
        },
        "candidate_positions": [[0, 0, 1]],
        "candidate_weights": [[1, 1, 1]],
        "candidate_refined_positions": [[1, 2, 1]],
    }
    figure = build_figure(load_scene(), result)
    names = [trace.name for trace in figure.data]
    assert "Predicted position" in names
    assert "Learned refinement" in names
    assert "Synthetic ground truth" not in names
    assert "Position error" not in names
    assert not any("Predicted position orientation" in name for name in names)


@pytest.mark.parametrize("color_mode", ["rgb", "feature"])
def test_real_pytorch3d_pointcloud_and_adapter(color_mode):
    pytest.importorskip(
        "pytorch3d.structures", reason="Optional PyTorch3D is not installed", exc_type=ImportError
    )
    scene = load_scene()
    cloud = to_pytorch3d_pointcloud(scene, color_mode)
    assert tuple(cloud.points_packed().shape) == (len(scene["points"]["xyz"]), 3)
    assert tuple(cloud.features_packed().shape) == (len(scene["points"]["xyz"]), 3)
    assert float(cloud.features_packed().min()) >= 0
    assert float(cloud.features_packed().max()) <= 1
    actual = build_figure(scene, backend="pytorch3d", color_mode=color_mode)
    portable = build_figure(scene, color_mode=color_mode)
    assert list(actual.data[0].x) == pytest.approx(list(portable.data[0].x), abs=1e-6)
    assert list(actual.data[0].y) == pytest.approx(list(portable.data[0].y), abs=1e-6)
    assert list(actual.data[0].z) == pytest.approx(list(portable.data[0].z), abs=1e-6)
    assert actual.layout.meta["backend"] == "pytorch3d"


def test_real_pytorch3d_quaternion_convention():
    pytest.importorskip(
        "pytorch3d.transforms", reason="Optional PyTorch3D is not installed", exc_type=ImportError
    )
    q = [0.5, 0.5, 0.5, 0.5]
    expected = quaternion_matrix(q)
    actual = pytorch3d_quaternion_matrix(q)
    for a, b in zip(actual, expected):
        assert a == pytest.approx(b, abs=1e-6)
    # The camera optical axis +Z is world +X for this pose convention.
    assert [actual[i][2] for i in range(3)] == pytest.approx([1, 0, 0])


def test_optional_full_pytorch3d_plot_scene():
    pytest.importorskip(
        "pytorch3d._C", reason="Full PyTorch3D compiled extension is optional", exc_type=ImportError
    )
    figure = build_full_pytorch3d_figure(load_scene())
    names = [trace.name for trace in figure.data]
    assert "RGB point cloud" in names
    assert "Trajectory cameras" in names
    assert figure.layout.meta["backend"] == "pytorch3d-plot_scene"
