import torch

from gnn_neo.nn.qat.modules import _get_qconfig, _get_fake_quantize


class Conv1d(torch.nn.Conv1d):
    def __init__(self, *args, **kwargs):
        qconfig = _get_qconfig(kwargs)
        super(Conv1d, self).__init__(*args, **kwargs)

        self.input_quantize, self.weight_quantize, self.bias_quantize, self.output_quantize = \
            _get_fake_quantize(qconfig)

    def forward(self, input):
        if self.bias is not None:
            return self.output_quantize(
                self._conv_forward(self.input_quantize(input), self.weight_quantize(self.weight),
                                   self.bias_quantize(self.bias)))
        else:
            return self.output_quantize(
                self._conv_forward(self.input_quantize(input), self.weight_quantize(self.weight), self.bias))


class Conv2d(torch.nn.Conv2d):
    def __init__(self, *args, **kwargs):
        qconfig = _get_qconfig(kwargs)
        # print(*args, **kwargs)
        super(Conv2d, self).__init__(*args, **kwargs)

        self.input_quantize, self.weight_quantize, self.bias_quantize, self.output_quantize = \
            _get_fake_quantize(qconfig)

    def forward(self, input):
        if self.bias is not None:
            return self.output_quantize(
                self._conv_forward(self.input_quantize(input), self.weight_quantize(self.weight),
                                   self.bias_quantize(self.bias)))
        else:
            return self.output_quantize(
                self._conv_forward(self.input_quantize(input), self.weight_quantize(self.weight), self.bias))


class Conv3d(torch.nn.Conv3d):
    def __init__(self, *args, **kwargs):
        qconfig = _get_qconfig(kwargs)
        super(Conv3d, self).__init__(*args, **kwargs)

        self.input_quantize, self.weight_quantize, self.bias_quantize, self.output_quantize = \
            _get_fake_quantize(qconfig)

    def forward(self, input):
        if self.bias is not None:
            return self.output_quantize(
                self._conv_forward(self.input_quantize(input), self.weight_quantize(self.weight),
                                   self.bias_quantize(self.bias)))
        else:
            return self.output_quantize(
                self._conv_forward(self.input_quantize(input), self.weight_quantize(self.weight), self.bias))
