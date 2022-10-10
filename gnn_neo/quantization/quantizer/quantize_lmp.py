import torch.nn
from gnn_neo.quantization.quantizer.mapping.nn_to_lmp import linear_to_lmp, conv1d_to_lmp, conv2d_to_lmp, conv3d_to_lmp
from gnn_neo.quantization.qconfig import QConfig, QConfigEntry, QScheme
from gnn_neo.quantization.quantizer.quantizer_base import QuantizerBase


def _get_default_mapping():
    mapping = {}
    mapping[torch.nn.Linear] = linear_to_lmp
    mapping[torch.nn.Conv1d] = conv1d_to_lmp
    mapping[torch.nn.Conv2d] = conv2d_to_lmp
    mapping[torch.nn.Conv3d] = conv3d_to_lmp
    return mapping


def _get_default_qconfig():
    def _get_qconfig_entry_tuple():
        qconfig_entry_2 = QConfigEntry(q_scheme=QScheme(bit=2))
        qconfig_entry_4 = QConfigEntry(q_scheme=QScheme(bit=4))
        qconfig_entry_6 = QConfigEntry(q_scheme=QScheme(bit=6))
        qconfig_entry_8 = QConfigEntry(q_scheme=QScheme(bit=8))
        return qconfig_entry_2, qconfig_entry_4, qconfig_entry_6, qconfig_entry_8
    weight_qconfig = _get_qconfig_entry_tuple()
    input_qconfig = _get_qconfig_entry_tuple()
    bias_qconfig = _get_qconfig_entry_tuple()
    return QConfig(input=input_qconfig, weight=weight_qconfig, bias=bias_qconfig)


class LayerWiseMultiPrecisionQuantizer(QuantizerBase):
    def __init__(self, qconfig=None):
        super(LayerWiseMultiPrecisionQuantizer, self).__init__(qconfig=qconfig)
        self.default_qconfig = _get_default_qconfig()
        self.mapping = _get_default_mapping()
