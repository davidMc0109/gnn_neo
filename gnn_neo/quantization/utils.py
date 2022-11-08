from gnn_neo.quantization.fake_quantize.quantize_base import QuantizeBase
import torch


def enable_calibration(model, enabled=True):
    for submodule in model.modules():
        if isinstance(submodule, QuantizeBase):
            assert hasattr(submodule, 'calibration')
            submodule.calibration.data = torch.tensor(enabled)


def enable_quantization(model, enabled=True):
    for submodule in model.modules():
        if isinstance(submodule, QuantizeBase):
            assert hasattr(submodule, 'quantization')
            submodule.quantization.data = torch.tensor(enabled)


def disable_calibration(model):
    enable_calibration(model, False)


def disable_quantization(model):
    enable_quantization(model, False)


def get_mp_search_modules(model):
    from gnn_neo.nn.qat.fast_layerwise_mixprecision.linear import Linear as FLMPLinear
    from gnn_neo.nn.qat.fast_layerwise_mixprecision.conv import Conv1d as FLMPConv1d
    from gnn_neo.nn.qat.fast_layerwise_mixprecision.conv import Conv2d as FLMPConv2d
    from gnn_neo.nn.qat.fast_layerwise_mixprecision.conv import Conv3d as FLMPConv3d
    from gnn_neo.nn.qat.layerwise_mixprecision.linear import Linear as LMPLinear
    from gnn_neo.nn.qat.layerwise_mixprecision.conv import Conv1d as LMPConv1d
    from gnn_neo.nn.qat.layerwise_mixprecision.conv import Conv2d as LMPConv2d
    from gnn_neo.nn.qat.layerwise_mixprecision.conv import Conv3d as LMPConv3d
    typeList = [FLMPLinear, FLMPConv1d, FLMPConv2d, FLMPConv3d, LMPLinear, LMPConv1d, LMPConv2d, LMPConv3d]
    ret = dict()
    for name, submodule in model.named_modules():
        for type in typeList:
            if isinstance(submodule, type):
                ret[name] = submodule
    return ret


def get_mp_params_size(model):
    ret = dict()
    mp_modules = get_mp_search_modules(model)
    for name, module in mp_modules.items():
        iq, wq, bq, oq = module.input_quantize, module.weight_quantize, module.bias_quantize, module.output_quantize
        assert not isinstance(wq, torch.nn.Identity)
        if not isinstance(iq, torch.nn.Identity):
            assert len(wq) == len(iq)
        if not isinstance(bq, torch.nn.Identity):
            assert len(wq) == len(bq)
        if not isinstance(oq, torch.nn.Identity):
            assert len(wq) == len(oq)
        w_size, b_size = list(), list()
        for wqq in wq:
            if isinstance(wqq, QuantizeBase):
                bw = wqq.qscheme.bit
                w_size.append(bw * module.weight.numel())
            else:
                w_size.append(32 * module.weight.numel())
        if isinstance(bq, torch.nn.Identity):
            for w in w_size:
                b_size.append(0)
        else:
            for bqq in bq:
                if isinstance(bqq, QuantizeBase):
                    bw = bqq.qscheme.bit
                    if module.bias is not None:
                        b_size.append(bw * module.bias.numel())
                    else:
                        b_size.append(0)
                else:
                    if module.bias is not None:
                        b_size.append(32 * module.bias.numel)
                    else:
                        b_size.append(0)
        # TODO: try to use real input/output size
        ret[name] = {
            'input_size': w_size,
            'weight_size': w_size,
            'bias_size': b_size,
            'output_size': w_size
        }
    return ret


def get_mp_alphas(model):
    ret = dict()
    mp_modules = get_mp_search_modules(model)
    for name, module in mp_modules.items():
        ret[name] = module.alpha
    return ret


def get_mp_params_cost(model):
    sizes = get_mp_params_size(model)
    alphas = get_mp_alphas(model)
    cost = torch.zeros((1,))
    for j, key in enumerate(sizes):
        alpha = alphas[key]
        normalized_alpha = torch.softmax(alpha, dim=-1)
        size = sizes[key]
        for i, value in enumerate(size.values()):
            if j == 0 and i == 0:
                cost = cost.to(normalized_alpha.device)
            cost += (normalized_alpha * torch.tensor(value, dtype=normalized_alpha.dtype,
                                                     device=normalized_alpha.device)).sum()
    return cost.reshape(())


def get_mp_weights(model):
    alphas = get_mp_alphas(model).values()
    weights = list(set(model.parameters()).difference(alphas))
    return weights
