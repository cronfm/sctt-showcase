import copy

import pytest
from pydantic import ValidationError
import torch

from sctt_showcase.data import get_sample, load_scene
from sctt_showcase.model import DemoEngine
from sctt_showcase.schemas import InferRequest


@pytest.fixture(scope="module")
def engine():
    torch.set_num_threads(2)
    return DemoEngine()


def test_bundled_samples_use_trained_model(engine):
    for sample in load_scene()["samples"]:
        result = engine.infer({"sample_id": sample["id"], "top_k": 4})
        assert result["model"]["status"] == "synthetic-trained"
        assert result["error_m"] < 0.5
        weights = torch.tensor(result["candidate_weights"])
        assert torch.allclose(weights.sum(0), torch.ones(3), atol=1e-6)
        assert result["truth"]["position"] == sample["position"]


def test_prediction_is_deterministic_and_candidate_offset_is_applied(engine):
    first = engine.infer({"sample_id": "loop-01"})
    second = engine.infer({"sample_id": "loop-01"})
    assert first["position"] == second["position"]
    offset = engine.infer({"sample_id": "loop-01", "candidate_offset_m": [0.1, 0, 0]})
    assert offset["candidate_positions"][0][0] == pytest.approx(first["candidate_positions"][0][0] + 0.1)


def test_custom_features_are_supported(engine):
    result = engine.infer(
        {
            "features": get_sample("loop-01")["features"],
            "candidates": [{k: v for k, v in c.items() if k != "score"} for c in load_scene()["candidates"]],
            "top_k": 2,
        }
    )
    assert result["sample_id"] == "custom"
    assert result["ground_truth"] is None
    assert len(result["candidate_ids"]) == 2


def test_masked_descriptor_values_do_not_change_prediction(engine):
    features = copy.deepcopy(get_sample("loop-01")["features"])
    features["mask"][-1] = False
    candidates = [{k: v for k, v in c.items() if k != "score"} for c in load_scene()["candidates"]]
    baseline = engine.infer({"features": features, "candidates": candidates})
    features["desc"][-1] = [500.0] * 16
    result = engine.infer({"features": features, "candidates": candidates})
    assert result["position"] == pytest.approx(baseline["position"], abs=1e-6)


@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_nonfinite_inputs_rejected(bad):
    with pytest.raises(ValidationError):
        InferRequest.model_validate({"candidate_offset_m": [bad, 0, 0]})


def test_empty_mask_and_mismatched_features_rejected():
    features = copy.deepcopy(get_sample("loop-01")["features"])
    features["mask"] = [False] * len(features["mask"])
    candidates = [{k: v for k, v in c.items() if k != "score"} for c in load_scene()["candidates"]]
    with pytest.raises(ValidationError):
        InferRequest.model_validate({"features": features, "candidates": candidates})
    features["mask"][0] = True
    features["uv"].pop()
    with pytest.raises(ValidationError):
        InferRequest.model_validate({"features": features, "candidates": candidates})


def test_bounds_and_custom_pairs_rejected():
    for request in ({"top_k": 0}, {"top_k": 9}, {"features": get_sample("loop-01")["features"]}):
        with pytest.raises(ValidationError):
            InferRequest.model_validate(request)
