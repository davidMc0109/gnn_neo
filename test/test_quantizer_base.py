from gnn_neo.quantization.quantizer.quantizer_base import QuantizerBase
from gnn_neo.quantization.quantizer.quantize_lmp import LayerWiseMultiPrecisionQuantizer
from gnn_neo.quantization.quantizer.quantize_flmp import FastLayerWiseMultiPrecisionQuantizer
from torchvision.models import resnet18

model = resnet18(pretrained=True)
# qmodel = QuantizerBase()(resnet18(pretrained=True))
# qmodel = LayerWiseMultiPrecisionQuantizer()(resnet18(pretrained=True))
qmodel = FastLayerWiseMultiPrecisionQuantizer()(resnet18(pretrained=True))
print(qmodel)

import torch
x = torch.randn(1, 3, 224, 224)
qy = qmodel(x)
y = model(x)
qy.sum().backward()
y.sum().backward()

hook = 0
