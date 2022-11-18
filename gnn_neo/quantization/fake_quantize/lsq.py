from typing import Any

from gnn_neo.quantization.fake_quantize.quantize_base import QuantizeBase
import torch


def grad_scale(t, scale):
    return (t - (t * scale)).detach() + (t * scale)


def _fake_quantize_learnable_per_channel_affine_training(x, scale, zero_point, ch_axis, quant_min, quant_max,
                                                         grad_factor):
    zero_point = (zero_point.round() - zero_point).detach() + zero_point
    new_shape = [1] * len(x.shape)
    new_shape[ch_axis] = x.shape[ch_axis]
    scale = grad_scale(scale, grad_factor).reshape(new_shape)
    zero_point = grad_scale(zero_point, grad_factor).reshape(new_shape)
    x = x / scale + zero_point
    x = (x.round() - x).detach() + x
    x = torch.clamp(x, quant_min, quant_max)
    return (x - zero_point) * scale


def _fake_quantize_learnable_per_tensor_affine_training(x, scale, zero_point, quant_min, quant_max, grad_factor):
    zero_point = (zero_point.round() - zero_point).detach() + zero_point
    scale = grad_scale(scale, grad_factor)
    zero_point = grad_scale(zero_point, grad_factor)
    x = x / scale + zero_point
    x = (x.round() - x).detach() + x
    x = torch.clamp(x, quant_min, quant_max)
    return (x - zero_point) * scale


class LearnableFakeQuantize(QuantizeBase):
    def __init__(self, qscheme, observer, use_grad_scaling=True):
        super(LearnableFakeQuantize, self).__init__(qscheme, observer)
        self.use_grad_scaling = use_grad_scaling
        self.register_buffer('scale', torch.tensor([1.0], dtype=torch.float))
        self.register_buffer('zero_point', torch.tensor([0], dtype=torch.int))

    def _quantization_impl(self, x):
        if self.use_grad_scaling:
            grad_factor = 1.0 / (x.numel() * self.qscheme.quant_max) ** 0.5
        else:
            grad_factor = 1.0
        if self.qscheme.per_channel:
            if 'ch_axis' in self.qscheme:
                ch_axis = self.qscheme.ch_axis
            else:
                ch_axis = 0
            x = _fake_quantize_learnable_per_channel_affine_training(x, self.scale, self.zero_grad, ch_axis,
                                                                     self.qscheme.quant_min, self.qscheme.quant_max,
                                                                     grad_factor)
        else:
            x = _fake_quantize_learnable_per_tensor_affine_training(x, self.scale, self.zero_grad,
                                                                    self.qscheme.quant_min, self.qscheme.quant_max,
                                                                    grad_factor)
        return x

    def _calibration_impl(self, x):
        self.scale, self.zero_grad = self.observer(x, self.qscheme)
