from gnn_neo.nn.qat.fast_layerwise_mixprecision.linear import Linear
from gnn_neo.nn.qat.fast_layerwise_mixprecision.conv import Conv1d, Conv2d, Conv3d


def _transfer_weights(source, target):
    target.load_state_dict(source.state_dict(), strict=False)


def linear_to_flmp(source, qconfig):
    ret = Linear(
        source.in_features,
        source.out_features,
        source.bias is not None,
        device=source.weight.device,
        dtype=source.weight.dtype,
        qconfig=qconfig)
    _transfer_weights(source, ret)
    return ret


def conv1d_to_flmp(source, qconfig):
    ret = Conv1d(
        source.in_channels,
        source.out_channels,
        source.kernel_size,
        source.stride,
        source.padding,
        source.dilation,
        source.groups,
        source.bias is not None,
        source.padding_mode,
        source.weight.device,
        source.weight.dtype,
        qconfig=qconfig
    )
    _transfer_weights(source, ret)
    return ret


def conv2d_to_flmp(source, qconfig):
    ret = Conv2d(
        source.in_channels,
        source.out_channels,
        source.kernel_size,
        source.stride,
        source.padding,
        source.dilation,
        source.groups,
        source.bias is not None,
        source.padding_mode,
        source.weight.device,
        source.weight.dtype,
        qconfig=qconfig
    )
    _transfer_weights(source, ret)
    return ret


def conv3d_to_flmp(source, qconfig):
    ret = Conv3d(
        source.in_channels,
        source.out_channels,
        source.kernel_size,
        source.stride,
        source.padding,
        source.dilation,
        source.groups,
        source.bias is not None,
        source.padding_mode,
        source.weight.device,
        source.weight.dtype,
        qconfig=qconfig
    )
    _transfer_weights(source, ret)
    return ret
