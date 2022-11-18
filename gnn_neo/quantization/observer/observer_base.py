import torch.nn as nn
from gnn_neo.quantization.qconfig import QScheme

class ObserverBase(nn.Module):
    def __init__(self):
        super(ObserverBase, self).__init__()
        # TODO:

    def forward(self, x, qscheme):
        assert isinstance(qscheme, QScheme)
        return self._observer_impl(x, qscheme)

    def _observer_impl(self, x, qscheme):
        import os
        if 'DEBUG' not in os.environ:
            raise NotImplementedError()
