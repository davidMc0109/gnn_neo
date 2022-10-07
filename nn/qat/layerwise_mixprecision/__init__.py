import torch

from ..utils import _get_qconfig as _general_get_qconfig
from quantization import QConfigEntry


def _get_qconfig(kwargs):
    branches = -1
    _CHECK_FIELDS = ['weight', 'bias', 'input', 'output']
    qconfig = _general_get_qconfig(kwargs)
    for field in _CHECK_FIELDS:
        checkee = qconfig[field]
        if checkee is not None:
            assert isinstance(checkee, tuple)
            if branches > 0:
                assert branches == len(checkee)
            branches = len(checkee)
            for entry in checkee:
                assert isinstance(entry, QConfigEntry)
    assert branches > 0
    return qconfig, branches

def _get_fake_quantize(qconfig):
    _, branches = _get_qconfig({'qconfig': qconfig})
    if qconfig.weight is not None:
        weight_quantize = list()
        for branch in qconfig.weight:
            weight_quantize.append(branch())
        weight_quantize = torch.nn.ModuleList(weight_quantize)
    else:
        weight_quantize = list()
        for branch in range(branches):
            weight_quantize.append(torch.nn.Identity())
        weight_quantize = torch.nn.ModuleList(weight_quantize)

    if qconfig.bias is not None:
        bias_quantize = list()
        for branch in qconfig.bias:
            bias_quantize.append(branch())
        bias_quantize = torch.nn.ModuleList(bias_quantize)
    else:
        bias_quantize = list()
        for branch in range(branches):
            bias_quantize.append(torch.nn.Identity())
        bias_quantize = torch.nn.ModuleList(bias_quantize)

    if qconfig.input is not None:
        input_quantize = list()
        for branch in qconfig.input:
            input_quantize.append(branch())
        input_quantize = torch.nn.ModuleList(input_quantize)
    else:
        input_quantize = list()
        for branch in range(branches):
            input_quantize.append(torch.nn.Identity())
        input_quantize = torch.nn.ModuleList(input_quantize)

    if qconfig.output is not None:
        output_quantize = list()
        for branch in qconfig.output:
            output_quantize.append(branch())
        output_quantize = torch.nn.ModuleList(output_quantize)
    else:
        output_quantize = list()
        for branch in range(branches):
            output_quantize.append(torch.nn.Identity())
        output_quantize = torch.nn.ModuleList(output_quantize)

    return input_quantize, weight_quantize, bias_quantize, output_quantize
