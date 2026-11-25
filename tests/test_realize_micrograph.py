import json

import pytest
import torch
from torch import nn

from capp_r.contracts.candidate_preflight import normalized_output_divergence
from capp_r.generator.realize_micrograph import (
    boundary_shape,
    build_module,
    parameter_count,
    replace_module,
)
from capp_r.grammar.ffn_grammar import enumerate_ffn
from capp_r.grammar.spatial_grammar import enumerate_spatial
from capp_r.grammar.type_system import TensorType


BOUNDARIES = [
    # (input, output): same shape, channel change, stride-2 downsampling, token (FFN) regions
    (TensorType("spatial", 64, 32, 32), TensorType("spatial", 64, 32, 32)),
    (TensorType("spatial", 128, 16, 16), TensorType("spatial", 64, 16, 16)),
    (TensorType("spatial", 32, 32, 32), TensorType("spatial", 64, 16, 16)),
    (TensorType("token", 96), TensorType("token", 96)),
    (TensorType("token", 64), TensorType("token", 128)),
]


@pytest.mark.parametrize("source,output", BOUNDARIES)
def test_all_grammar_graphs_realize_with_exact_boundaries_and_parameter_estimates(source, output):
    generator = enumerate_spatial if source.kind == "spatial" else enumerate_ffn
    graphs = list(generator(source, output))
    assert graphs, "grammar produced no candidates for a supported boundary"
    for graph_spec in graphs:
        graph = json.loads(json.dumps(graph_spec.to_dict()))  # the serialized form used in manifests
        module = build_module(graph).eval()
        value = torch.randn(boundary_shape(graph["input_type"]))
        with torch.no_grad():
            output = module(value)
        assert tuple(output.shape) == boundary_shape(graph["output_type"])
        assert parameter_count(module) == int(graph["estimated_parameters"])


def test_replace_module_returns_original_and_installs_replacement():
    model = nn.Sequential(nn.Identity(), nn.Sequential(nn.ReLU()))
    replacement = nn.SiLU()
    original = replace_module(model, "1.0", replacement)
    assert isinstance(original, nn.ReLU)
    assert model[1][0] is replacement


def test_unknown_operator_fails_closed():
    spec = {
        "op": "UnapprovedOperator",
        "input_type": {"kind": "token", "channels": 8},
        "output_type": {"kind": "token", "channels": 8},
        "attributes": {},
        "children": [],
    }
    with pytest.raises(ValueError, match="unsupported micrograph operator"):
        build_module(spec)


def test_normalized_output_divergence_compares_matching_outputs():
    parent = {"pred_boxes": torch.ones(1, 2, 4)}
    child = {"pred_boxes": torch.full((1, 2, 4), 1.25)}
    assert normalized_output_divergence(parent, child) == pytest.approx(0.25)
