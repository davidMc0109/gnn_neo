import torch

from ..utils import _get_qconfig as _general_get_qconfig
from quantization import QConfigEntry

def _get_qconfig(kwargs):
    _CHECK_FIELDS = ['weight', 'bias', 'input', 'output']
    qconfig = _general_get_qconfig(kwargs)
    for field in _CHECK_FIELDS:
        checkee = qconfig[field]
        assert checkee is None or isinstance(checkee, QConfigEntry)
    return qconfig


def _get_fake_quantize(qconfig):
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
