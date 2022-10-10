import torch.nn
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


class QuantizerBase:
    def __init__(self, qconfig=None, mapping=None):
        super(QuantizerBase, self).__init__()
        if qconfig is None:
            qconfig = {}
        self.mapping = _get_default_mapping() if mapping is None else mapping
        self.qconfig = qconfig
        self.default_qconfig = _get_default_qconfig()

    def _apply_quant(self, ctx, prefix=''):
        for name, module in ctx.named_children():
            module = self._apply_quant(module, prefix+'.'+name)
            for k in self.mapping:
                if isinstance(module, k):
                    if prefix+'.'+name in self.qconfig:
                        module = self.mapping[k](module, self.qconfig[prefix+'.'+name])
                    else:
                        module = self.mapping[k](module, self.default_qconfig)
            ctx._modules[name] = module
        return ctx

    def register_mapping(self, name, convert_fn):
        self.mapping[name] = convert_fn

    def __call__(self, model):
        return self._apply_quant(model)
