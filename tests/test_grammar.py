from capp_r.generator.graph_hash import graph_hash
from capp_r.grammar.combinators import parallel_sum, sequential
from capp_r.grammar.primitives import conv, identity
from capp_r.grammar.type_system import TensorType, assert_boundary


def test_stride_is_contract_typed():
    source = TensorType("spatial", 64, 80, 80)
    hidden = conv(source, 32, 1, stride=2)
    graph = sequential(hidden, conv(hidden.output_type, 128, 1))
    assert_boundary(graph, source, TensorType("spatial", 128, 40, 40))


def test_parallel_sum_hash_is_commutative():
    source = TensorType("spatial", 64, 40, 40)
    left = identity(source)
    right = conv(source, 64, 1)
    assert graph_hash(parallel_sum(left, right)) == graph_hash(parallel_sum(right, left))


def test_invalid_boundary_is_rejected():
    source = TensorType("spatial", 64, 40, 40)
    graph = conv(source, 32, 1)
    try:
        assert_boundary(graph, source, source)
    except TypeError:
        return
    raise AssertionError("mismatched output channels were accepted")
