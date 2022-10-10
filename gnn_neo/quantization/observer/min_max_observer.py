import torch

from gnn_neo.quantization.observer.observer_base import ObserverBase

class MinMaxObserver(ObserverBase):
    def __init__(self):
        super(MinMaxObserver, self).__init__()

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
        scale = (qscheme.quant_max-qscheme.quant_min) / (max_val_cur-min_val_cur)
        scale = torch.log2(scale)
        if qscheme.pot_scale:
            scale = torch.ceil(scale)
        zero_point = (max_val_cur+min_val_cur)*scale/2
        scale = torch.pow(2, -scale)
        return scale, zero_point


