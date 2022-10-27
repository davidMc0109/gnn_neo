from gnn_neo.quantization.quantizer.quantizer_base import QuantizerBase as TransformationBase
from gnn_neo.quantization.quantizer.mapping.pyg_to_torch import linear_pyg_to_torch
import torch
from torch_geometric.nn import Linear as PygLinear

def _get_default_mapping():
    mapping = {}
    mapping[PygLinear] = linear_pyg_to_torch
    return mapping

class PygToTorchTransformer(TransformationBase):
    def __init__(self):
        super(PygToTorchTransformer, self).__init__(qconfig=None)
        self.default_qconfig = None
        self.mapping = _get_default_mapping()
