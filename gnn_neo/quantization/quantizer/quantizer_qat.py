import torch.nn
from gnn_neo.quantization.quantizer.quantizer_base import QuantizerBase
from gnn_neo.quantization.quantizer.mapping.nn_to_qat import linear_to_qat, conv1d_to_qat, conv2d_to_qat, conv3d_to_qat
from gnn_neo.quantization.qconfig import QConfig


def _get_default_mapping():
    mapping = {}
    mapping[torch.nn.Linear] = linear_to_qat
    mapping[torch.nn.Conv1d] = conv1d_to_qat
    mapping[torch.nn.Conv2d] = conv2d_to_qat
    mapping[torch.nn.Conv3d] = conv3d_to_qat

    return mapping


def _get_default_qconfig():
    return QConfig()


class QATQuantizer(QuantizerBase):
    def __init__(self, qconfig=None, mapping=None):
        super(QATQuantizer, self).__init__(qconfig=qconfig)
        self.mapping = _get_default_mapping() if mapping is None else mapping
        self.default_qconfig = _get_default_qconfig()

