import torch

from . import _get_qconfig, _get_fake_quantize


class Conv1d(torch.nn.Conv1d):
    def __init__(self, *args, **kwargs):
        qconfig, num_branches = _get_qconfig(kwargs)
        super(Conv1d, self).__init__(*args, **kwargs)

        self.input_quantize, self.weight_quantize, self.bias_quantize, self.output_quantize = \
            _get_fake_quantize(qconfig)
        self.alpha = torch.nn.Parameter(
            torch.randn((num_branches,), dtype=self.weight.dtype, device=self.weight.device))

    def forward(self, input):
        normalized_alpha = self.alpha.norm()
        weights, inputs, biases = None, None, None
        for branch, alpha in enumerate(normalized_alpha):
            weight = self.weight_quantize[branch](self.weight)
            input = self.input_quantize[branch](input)
            if weights is None:
                weights = torch.zeros((len(normalized_alpha),) + weight.shape, dtype=weight.dtype, device=weight.device)
            if inputs is None:
                inputs = torch.zeros((len(normalized_alpha),) + input.shape, dtype=input.dtype, device=input.device)
            weights[branch, ...] = weight
            inputs[branch, ...] = input
            if self.bias is not None:
                bias = alpha * self.bias_quantize[branch](self.bias)
                if biases is None:
                    biases = torch.zeros((len(bias),) + bias.shape, dtype=bias.dtype, device=bias.device)
                biases[branch, ...] = bias[...]
        weight = torch.sum(torch.mul(weights, normalized_alpha), dim=0)
        input = torch.sum(torch.mul(inputs, normalized_alpha), dim=0)
        if biases is not None:
            bias = torch.sum(torch.mul(biases, normalized_alpha), dim=0)
        else:
            bias = None

        return self.output_quantize(self._conv_forward(input, weight, bias))

class Conv2d(torch.nn.Conv1d):
    def __init__(self, *args, **kwargs):
        qconfig, num_branches = _get_qconfig(kwargs)
        super(Conv2d, self).__init__(*args, **kwargs)

        self.input_quantize, self.weight_quantize, self.bias_quantize, self.output_quantize = \
            _get_fake_quantize(qconfig)
        self.alpha = torch.nn.Parameter(
            torch.randn((num_branches,), dtype=self.weight.dtype, device=self.weight.device))

    def forward(self, input):
        normalized_alpha = self.alpha.norm()
        weights, inputs, biases = None, None, None
        for branch, alpha in enumerate(normalized_alpha):
            weight = self.weight_quantize[branch](self.weight)
            input = self.input_quantize[branch](input)
            if weights is None:
                weights = torch.zeros((len(normalized_alpha),) + weight.shape, dtype=weight.dtype, device=weight.device)
            if inputs is None:
                inputs = torch.zeros((len(normalized_alpha),) + input.shape, dtype=input.dtype, device=input.device)
            weights[branch, ...] = weight
            inputs[branch, ...] = input
            if self.bias is not None:
                bias = alpha * self.bias_quantize[branch](self.bias)
                if biases is None:
                    biases = torch.zeros((len(bias),) + bias.shape, dtype=bias.dtype, device=bias.device)
                biases[branch, ...] = bias[...]
        weight = torch.sum(torch.mul(weights, normalized_alpha), dim=0)
        input = torch.sum(torch.mul(inputs, normalized_alpha), dim=0)
        if biases is not None:
            bias = torch.sum(torch.mul(biases, normalized_alpha), dim=0)
        else:
            bias = None

        return self.output_quantize(self._conv_forward(input, weight, bias))

class Conv3d(torch.nn.Conv1d):
    def __init__(self, *args, **kwargs):
        qconfig, num_branches = _get_qconfig(kwargs)
        super(Conv3d, self).__init__(*args, **kwargs)

        self.input_quantize, self.weight_quantize, self.bias_quantize, self.output_quantize = \
            _get_fake_quantize(qconfig)
        self.alpha = torch.nn.Parameter(
            torch.randn((num_branches,), dtype=self.weight.dtype, device=self.weight.device))

    def forward(self, input):
        normalized_alpha = self.alpha.norm()
        weights, inputs, biases = None, None, None
        for branch, alpha in enumerate(normalized_alpha):
            weight = self.weight_quantize[branch](self.weight)
            input = self.input_quantize[branch](input)
            if weights is None:
                weights = torch.zeros((len(normalized_alpha),)+weight.shape, dtype=weight.dtype, device=weight.device)
            if inputs is None:
                inputs = torch.zeros((len(normalized_alpha),)+input.shape, dtype=input.dtype, device=input.device)
            weights[branch, ...] = weight
            inputs[branch, ...] = input
            if self.bias is not None:
                bias = alpha*self.bias_quantize[branch](self.bias)
                if biases is None:
                    biases = torch.zeros((len(bias),)+bias.shape, dtype=bias.dtype, device=bias.device)
                biases[branch, ...] = bias[...]
        weight = torch.sum(torch.mul(weights, normalized_alpha), dim=0)
        input = torch.sum(torch.mul(inputs, normalized_alpha), dim=0)
        if biases is not None:
            bias = torch.sum(torch.mul(biases, normalized_alpha), dim=0)
        else:
            bias = None

        return self.output_quantize(self._conv_forward(input, weight, bias))

