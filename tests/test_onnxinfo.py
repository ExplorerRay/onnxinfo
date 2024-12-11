import onnxinfo
import pytest

def test_read_onnx_nofile():
    with pytest.raises(ValueError):
        onnxinfo.read_onnx('non-existing.onnx')

def test_read_onnx():
    info = onnxinfo.read_onnx('models/resnet18_Opset16.onnx')
    assert info is not None

def test_infer_shape():
    model = onnxinfo.read_onnx('models/resnet18_Opset16.onnx')
    target = onnxinfo.InferShapeImpl(model.graph())
    old_size = len(target.get_ndname_to_shape())
    target.infer_shapes()
    new_size = len(target.get_ndname_to_shape())
    assert new_size > old_size

def test_print_summary():
    try:
        model = onnxinfo.read_onnx('models/resnet18_Opset16.onnx')
        target = onnxinfo.InferShapeImpl(model.graph())
        target.infer_shapes()
        target.print_summary()
    except:
        pytest.fail("iterate_graph failed")
