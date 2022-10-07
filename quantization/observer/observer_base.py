import torch.nn as nn

class ObserverBase(nn.Module):
    def __init__(self):
        super(ObserverBase, self).__init__()

    def forward(self, x, quantizer):
        assert isinstance(quantizer, ObserverBase)
        return self._observer_impl(x)

    def _observer_impl(self, x):
        import os
        if 'DEBUG' not in os.environ:
            raise NotImplementedError()
