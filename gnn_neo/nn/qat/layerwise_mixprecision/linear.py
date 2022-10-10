import torch
import torch.nn.functional as F

from gnn_neo.nn.qat.layerwise_mixprecision import _get_qconfig, _get_fake_quantize


class Linear(torch.nn.Linear):
    def __init__(self, *args, **kwargs):
        qconfig, num_branches = _get_qconfig(kwargs)
        super(Linear, self).__init__(*args, **kwargs)

        self.input_quantize, self.weight_quantize, self.bias_quantize, self.output_quantize = \
            _get_fake_quantize(qconfig)
        self.alpha = torch.nn.Parameter(torch.randn((num_branches,), dtype=self.weight.dtype, device=self.weight.device))

    def forward(self, input):
        normalized_alpha = torch.softmax(self.alpha, dim=-1)
        outputs = None
        for branch, alpha in enumerate(normalized_alpha):
            if self.bias is not None:
                output = self.output_quantize[branch](F.linear(self.input_quantize[branch](input), self.weight_quantize[branch](self.weight), self.bias_quantize[branch](self.bias)))
            else:
                output = self.output_quantize[branch](F.linear(self.input_quantize[branch](input), self.weight_quantize[branch](self.weight)))
            if outputs is None:
                outputs = torch.zeros((len(normalized_alpha),)+output.shape, dtype=output.dtype, device=output.device)
            outputs[branch, ...] = normalized_alpha[branch] * output[...]
        output = torch.sum(outputs, 0)
        return output

