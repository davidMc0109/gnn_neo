from gnn_neo.quantization.fake_quantize.quantize_base import QuantizeBase


def enable_calibration(model, enabled=True):
    for submodule in model.modules():
        if isinstance(submodule, QuantizeBase):
            assert hasattr(submodule, 'calibration')
            submodule.calibration = enabled


def enable_quantization(model, enabled=True):
    for submodule in model.modules():
        if isinstance(submodule, QuantizeBase):
            assert hasattr(submodule, 'quantization')
            submodule.quantization = enabled


def disable_calibration(model):
    enable_calibration(model, False)


def disable_quantization(model):
    enable_quantization(model, False)

