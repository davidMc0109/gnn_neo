import torch
import torch.nn as nn

from gnn_neo.quantization.qconfig import QScheme
from gnn_neo.quantization.observer.observer_base import ObserverBase


class QuantizeBase(nn.Module):
    def __init__(self, qscheme, observer):
        assert isinstance(qscheme, QScheme)
        assert isinstance(observer, ObserverBase)
        super(QuantizeBase, self).__init__()
        self.qscheme = qscheme
        self.observer = observer
        self.calibration = torch.tensor(True)
        self.quantization = torch.tensor(True)

    def forward(self, x):
        if self.calibration:
            self._calibration_impl(x)
        if self.quantization:
            x = self._quantization_impl(x)
        return x

    def _quantization_impl(self, x):
        import os
        if 'DEBUG' not in os.environ:
            raise NotImplementedError()

    def _calibration_impl(self, x):
        import os
        if 'DEBUG' not in os.environ:
            raise NotImplementedError()
