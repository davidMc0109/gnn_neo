import torch

from gnn_neo.nn.qat.utils import _get_qconfig as _general_get_qconfig
from gnn_neo.quantization.qconfig import QConfigEntry, QConfig

def _get_qconfig(kwargs):
    _CHECK_FIELDS = ['weight', 'bias', 'input', 'output']
    qconfig = _general_get_qconfig(kwargs)
    for field in _CHECK_FIELDS:
        checkee = qconfig[field]
        assert checkee is None or isinstance(checkee, QConfigEntry)
    return qconfig

def _check_qconfig(qconfig):
    """
    Checks if input is a valid QConfig object
    """
    assert isinstance(qconfig, QConfig)
    _CHECK_FIELDS = ['weight', 'bias', 'input', 'output']
    for field in _CHECK_FIELDS:
        checkee = qconfig[field]
        assert checkee is None or isinstance(checkee, QConfigEntry)
    return True

def _get_fake_quantize(qconfig: QConfig):
    """
    Break down qconfig into input, weight, output, and bias quantization
    configs
    :param qconfig:
    :return:
    """
    if qconfig.weight is not None:
        weight_quantize = qconfig.weight()
    else:
        weight_quantize = torch.nn.Identity()

    if qconfig.bias is not None:
        bias_quantize = qconfig.bias()
    else:
        bias_quantize = torch.nn.Identity()

    if qconfig.input is not None:
        input_quantize = qconfig.input()
    else:
        input_quantize = torch.nn.Identity()

    if qconfig.output is not None:
        output_quantize = qconfig.output()
    else:
        output_quantize = torch.nn.Identity()
    return input_quantize, weight_quantize, bias_quantize, output_quantize
