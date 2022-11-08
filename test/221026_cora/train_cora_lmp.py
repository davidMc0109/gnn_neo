import os.path as osp

import torch
import torch.nn.functional as F

import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import SplineConv

from gnn_neo.quantization.quantizer.quantize_lmp import LayerWiseMultiPrecisionQuantizer
from gnn_neo.quantization.quantizer.pyg_to_torch import PygToTorchTransformer
from gnn_neo.quantization.utils import get_mp_alphas, get_mp_params_cost


dataset = 'Cora'
transform = T.Compose([
    T.RandomNodeSplit(num_val=500, num_test=500),
    T.TargetIndegree(),
])
path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', dataset)
dataset = Planetoid(path, dataset, transform=transform)
data = dataset[0]


class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = SplineConv(dataset.num_features, 16, dim=1, kernel_size=2)
        self.conv2 = SplineConv(16, dataset.num_classes, dim=1, kernel_size=2)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        x = F.dropout(x, training=self.training)
        x = F.elu(self.conv1(x, edge_index, edge_attr))
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index, edge_attr)
        return F.log_softmax(x, dim=1)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model, data = Net().to(device), data.to(device)
model = PygToTorchTransformer()(model)
model = LayerWiseMultiPrecisionQuantizer()(model)
alphas = get_mp_alphas(model).values()
weights = list(set(model.parameters()).difference(alphas))
optimizer = torch.optim.Adam(weights, lr=0.01, weight_decay=5e-3)
optimizer_mp = torch.optim.Adam(alphas, lr=0.01)


def train():
    model.train()
    optimizer.zero_grad()
    optimizer_mp.zero_grad()
    acc_loss = F.nll_loss(model(data)[data.train_mask], data.y[data.train_mask])
    mp_loss = get_mp_params_cost(model)
    loss = acc_loss + 0.03*mp_loss * (acc_loss.detach().clone() / mp_loss.detach().clone())
    loss.backward()
    optimizer.step()
    optimizer_mp.step()


@torch.no_grad()
def test():
    model.eval()
    log_probs, accs = model(data), []
    for _, mask in data('train_mask', 'test_mask'):
        pred = log_probs[mask].max(1)[1]
        acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
        accs.append(acc)
    return accs


for epoch in range(1, 201):
    train()
    train_acc, test_acc = test()
    print(f'Epoch: {epoch:03d}, Train: {train_acc:.4f}, Test: {test_acc:.4f}')

hook = 1
