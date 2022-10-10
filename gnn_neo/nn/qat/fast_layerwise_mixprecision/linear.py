import torch
import torch.nn.functional as F

from gnn_neo.nn.qat.fast_layerwise_mixprecision import _get_qconfig, _get_fake_quantize


class Linear(torch.nn.Linear):
    def __init__(self, *args, **kwargs):
        qconfig, num_branches = _get_qconfig(kwargs)
        super(Linear, self).__init__(*args, **kwargs)

        self.input_quantize, self.weight_quantize, self.bias_quantize, self.output_quantize = \
            _get_fake_quantize(qconfig)
        self.alpha = torch.nn.Parameter(torch.randn((num_branches,), dtype=self.weight.dtype, device=self.weight.device))

    def forward(self, input):
        normalized_alpha = torch.softmax(self.alpha, dim=-1)
        weights, inputs, biases = None, None, None
        for branch, alpha in enumerate(normalized_alpha):
            weight = self.weight_quantize[branch](self.weight)
            input = self.input_quantize[branch](input)
            if weights is None:
                weights = torch.zeros((len(normalized_alpha),) + weight.shape, dtype=weight.dtype, device=weight.device)
            if inputs is None:
                inputs = torch.zeros((len(normalized_alpha),) + input.shape, dtype=input.dtype, device=input.device)
            weights[branch, ...] = normalized_alpha[branch] * weight
            inputs[branch, ...] = normalized_alpha[branch] * input
            if self.bias is not None:
                bias = self.bias_quantize[branch](self.bias)
                if biases is None:
                    biases = torch.zeros((len(bias),) + bias.shape, dtype=bias.dtype, device=bias.device)
                biases[branch, ...] = normalized_alpha[branch] * bias[...]
        weight = torch.sum(weights, dim=0)
        input = torch.sum(inputs, dim=0)
        if biases is not None:
            bias = torch.sum(biases, dim=0)
        else:
            bias = None
        return self.output_quantize(F.linear(input, weight, bias))

