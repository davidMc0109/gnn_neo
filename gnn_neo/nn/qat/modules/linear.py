import torch
import torch.nn.functional as F

from gnn_neo.nn.qat.modules import _get_qconfig, _get_fake_quantize


class Linear(torch.nn.Linear):
    def __init__(self, *args, **kwargs):
        qconfig = _get_qconfig(kwargs)
        super(Linear, self).__init__(*args, **kwargs)

        self.input_quantize, self.weight_quantize, self.bias_quantize, self.output_quantize = \
            _get_fake_quantize(qconfig)

    def forward(self, input):
        if self.bias is not None:
            return self.output_quantize(
                F.linear(self.input_quantize(input), self.weight_quantize(self.weight), self.bias_quantize(self.bias)))
        else:
            return self.output_quantize(F.linear(self.input_quantize(input), self.weight_quantize(self.weight)))


