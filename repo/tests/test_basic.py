import importlib, torch
def test_cuda_on():
    assert torch.cuda.is_available()
def test_pkg_import():
    importlib.import_module("ovod")