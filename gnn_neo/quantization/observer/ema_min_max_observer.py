import torch

from gnn_neo.quantization.observer.min_max_observer import MinMaxObserver


class EMAMinMaxObserver(MinMaxObserver):
    def __init__(self, ema_scale=0.8):
        super(EMAMinMaxObserver, self).__init__()
        self.register_buffer('ema_scale', torch.Tensor([ema_scale]))
        self.register_buffer('min_old', None)
        self.register_buffer('max_old', None)

    def _observer_impl(self, x, qscheme):
        x = x.detach()
        if qscheme.per_channel:
            if 'ch_axis' in qscheme:
                ch_axis = qscheme.ch_axis
            else:
                ch_axis = 0
            x_dims = x.size()
            new_axis_list = [i for i in range(len(x_dims))]
            new_axis_list[ch_axis] = 0
            new_axis_list[0] = ch_axis
            y = torch.flatten(x.permute(new_axis_list), start_dim=1)
            min_val_cur, max_val_cur = torch.aminmax(y, dim=1)
        else:
            min_val_cur, max_val_cur = torch.aminmax(x)
        if qscheme.symmetry:
            max_abs = min_val_cur.abs() if min_val_cur.abs() > max_val_cur.abs() else max_val_cur.abs()
            min_val_cur, max_val_cur = -max_abs, max_abs
        if self.min_old is None:
            self.min_old = min_val_cur
        if self.max_old is None:
            self.max_old = max_val_cur
        min_val_cur = self.ema_scale*self.min_old + (1-self.ema_scale)*min_val_cur
        max_val_cur = self.ema_scale*self.max_old + (1-self.ema_scale)*max_val_cur
        self.min_old, self.max_old = min_val_cur, max_val_cur

        scale = torch.log2(max_val_cur-min_val_cur) / (qscheme.quant_max-qscheme.quant_min)
        if qscheme.pot:
            scale = torch.ceil(scale)
        zero_point = (max_val_cur+min_val_cur)/(2*scale)
        return scale, zero_point

