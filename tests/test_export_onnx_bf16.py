from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from PipeLine.tools.export_onnx import convert_onnx_initializers_to_bf16


def test_bfloat16_initializer_storage_keeps_float_compute_contract(tmp_path: Path):
    weight = numpy_helper.from_array(
        np.array([[1.0, -2.5], [3.25, 4.0]], dtype=np.float32), name="weight"
    )
    graph = helper.make_graph(
        [helper.make_node("MatMul", ["input", "weight"], ["output"])],
        "toy",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 2])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 2])],
        [weight],
    )
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=10
    )
    source = tmp_path / "source.onnx"
    target = tmp_path / "target.onnx"
    onnx.save_model(model, source)

    assert convert_onnx_initializers_to_bf16(source, target) == 1
    converted = onnx.load(target)
    onnx.checker.check_model(converted)
    assert converted.graph.initializer[0].data_type == TensorProto.BFLOAT16
    assert converted.graph.initializer[0].name == "weight__bf16_storage"
    cast = converted.graph.node[0]
    assert cast.op_type == "Cast"
    assert list(cast.input) == ["weight__bf16_storage"]
    assert list(cast.output) == ["weight"]
    assert converted.graph.input[0].type.tensor_type.elem_type == TensorProto.FLOAT
    assert converted.graph.output[0].type.tensor_type.elem_type == TensorProto.FLOAT
