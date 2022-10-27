'''
Test file for channel-wise linear quantization
'''
import torch
import torch.nn as nn
from torch.nn import Linear
from gnn_neo.quantization.quantizer.quantizer_base import QuantizerBase
from gnn_neo.quantization.qconfig import QConfig, QScheme, QConfigEntry
from gnn_neo.quantization.utils import enable_quantization, enable_calibration


class TestModel(nn.Module):
    def __init__(self, in_features: int):
        super().__init__()

        self.lin = Linear(in_features, 2, bias=False)

    def forward(self, x):
        return self.lin(x)


# Create a simple linear model
model = TestModel(in_features=1)

# Set weights to emulate a pre-trained model
with torch.no_grad():
    model.lin.weight[0, 0] = 1.0
    model.lin.weight[1, 0] = 2.0
model.eval()

# Prepare quantizer
qconfig_lin0 = QConfig(
    input=QConfigEntry(q_scheme=QScheme(bit=8, symmetry=True)),
    weight=QConfigEntry(q_scheme=QScheme(bit=8, symmetry=True)),
    bias=QConfigEntry(q_scheme=QScheme(bit=8, symmetry=True)),
    output=QConfigEntry(q_scheme=QScheme(bit=8, symmetry=True)),
)
qconfig_lin1 = QConfig(
    input=QConfigEntry(q_scheme=QScheme(bit=6, symmetry=True)),
    weight=QConfigEntry(q_scheme=QScheme(bit=6, symmetry=True)),
    bias=QConfigEntry(q_scheme=QScheme(bit=6, symmetry=True)),
    output=QConfigEntry(q_scheme=QScheme(bit=6, symmetry=True)),
)

qconfig = {
    '.lin': {
        0: qconfig_lin0,
        1: qconfig_lin1,
    }
}

quantizer = QuantizerBase(qconfig=qconfig)

# Quantize model
qmodel = quantizer(model)

print(qmodel.lin)
