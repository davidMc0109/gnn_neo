from typing import Any

from gnn_neo.quantization.fake_quantize.quantize_base import QuantizeBase
import torch

def dsq_function_per_tensor(x, scale, zero_point, quant_min, quant_max, alpha):
    tanh_scale = 1 / (1 - alpha)
    tanh_k = math.log((tanh_scale + 1) / (tanh_scale - 1))

    x = x / scale + zero_point
    x = torch.clamp(x, quant_min, quant_max)
    x = x.floor() + (tanh_scale * torch.tanh(tanh_k * (x - x.floor() - 0.5))) * 0.5 + 0.5
    x = (x.round() - x).detach() + x
    x = (x - zero_point) * scale

    return x


def dsq_function_per_channel(x, scale, zero_point, quant_min, quant_max, ch_axis, alpha):

    new_shape = [1] * len(x.shape)
    new_shape[ch_axis] = x.shape[ch_axis]
    scale = scale.reshape(new_shape)
    zero_point = zero_point.reshape(new_shape)

    tanh_scale = 1 / (1 - alpha)
    tanh_k = math.log((tanh_scale + 1) / (tanh_scale - 1))

    x = x / scale + zero_point
    x = torch.clamp(x, quant_min, quant_max)
    x = x.floor() + (tanh_scale * torch.tanh(tanh_k * (x - x.floor() - 0.5))) * 0.5 + 0.5
    x = (x.round() - x).detach() + x
    x = (x - zero_point) * scale

    return x


class FakeQuantizeDSQPerchannel(torch.autograd.Function):
    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any) -> Any:
        pass

    @staticmethod
    def jvp(ctx: Any, *grad_inputs: Any) -> Any:
        pass

    @staticmethod
    def forward(ctx, x, scale, zero_point, quant_min, quant_max, ch_axis, alpha):
        return dsq_function_per_channel(x, scale, zero_point, quant_min, quant_max, ch_axis, alpha)

    @staticmethod
    def symbolic(g, x, scale, zero_point, quant_min, quant_max, ch_axis, alpha):
        return g.op("::FakeQuantizeDSQPerchannel", x, scale, zero_point, quant_min_i=quant_min, quant_max_i=quant_max, alpha_f=alpha)


class FakeQuantizeDSQPertensor(torch.autograd.Function):
    @staticmethod
    def jvp(ctx: Any, *grad_inputs: Any) -> Any:
        pass

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any) -> Any:
        pass

    @staticmethod
    def forward(ctx, x, scale, zero_point, quant_min, quant_max, alpha):
        return dsq_function_per_tensor(x, scale, zero_point, quant_min, quant_max, alpha)

    @staticmethod
    def symbolic(g, x, scale, zero_point, quant_min, quant_max, alpha):
        return g.op("::FakeQuantizeDSQPertensor", x, scale, zero_point, quant_min_i=quant_min, quant_max_i=quant_max, alpha_f=alpha)


class DSQFakeQuantize(QuantizeBase):
    def __init__(self, qscheme, observer, alpha=0.4):
        super(DSQFakeQuantize, self).__init__(qscheme, observer)
        self.register_buffer('scale', torch.tensor([1.0], dtype=torch.float))
        self.register_buffer('zero_point', torch.tensor([0], dtype=torch.int))
        self.qscheme = qscheme
        self.observer = observer
        self.alpha = torch.nn.Parameter(torch.tensor([alpha], dtype=torch.float))

    def _quantization_impl(self, x):
        self.scale, self.zero_point = self.scale.to(x.device), self.zero_point.to(x.device)
        if self.qscheme.per_channel:
            if 'ch_axis' in self.qscheme:
                ch_axis = self.qscheme.ch_axis
            else:
                ch_axis = 0
            x = dsq_function_per_channel(x, self.scale, self.zero_point, self.qscheme.quant_min, self.qscheme.quant_max, ch_axis, self.alpha)
        else:
            x = dsq_function_per_tensor(x, self.scale, self.zero_point, self.qscheme.quant_min, self.qscheme.quant_max, self.alpha)
        return x

    def _calibration_impl(self, x):
        self.scale, self.zero_point = self.observer(x, self.qscheme)
