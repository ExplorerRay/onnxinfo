import _onnxinfo
import pytest

def test_read_onnx_nofile():
    with pytest.raises(FileNotFoundError):
        _onnxinfo.read_onnx('non-existing.onnx')

def test_read_onnx():
    info = _onnxinfo.read_onnx('models/resnet50-new.onnx')
    assert info is not None

def test_iterate_onnx():
    info = _onnxinfo.read_onnx('models/resnet50-new.onnx')
    assert info is not None

    for node in info.nodes:
        assert node is not None
