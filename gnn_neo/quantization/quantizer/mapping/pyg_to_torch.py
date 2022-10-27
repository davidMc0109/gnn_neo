from torch.nn import Linear as TorchLinear
from torch_geometric.nn import Linear as PygLinear

def _transfer_weights(source, target):
    target.load_state_dict(source.state_dict(), strict=False)

def linear_pyg_to_torch(source, qconfig="FakeStub"):
    ret = TorchLinear(
        source.in_channels,
        source.out_channels,
        source.bias is not None,
        device=source.weight.device,
        dtype=source.weight.dtype,
        )
    _transfer_weights(source, ret)
    return ret
