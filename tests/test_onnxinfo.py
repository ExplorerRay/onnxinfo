import _onnxinfo
import pytest

def test_read_onnx_nofile():
    with pytest.raises(ValueError):
        _onnxinfo.read_onnx('non-existing.onnx')

def test_read_onnx():
    info = _onnxinfo.read_onnx('models/resnet50-new.onnx')
    assert info is not None

def test_iterate_onnx():
    try:
        graph = _onnxinfo.read_onnx('models/resnet50-new.onnx')
        _onnxinfo.iterate_graph(graph)
    except:
        pytest.fail("iterate_graph failed")
