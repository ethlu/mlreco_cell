import torch
from torch import nn
from models.cellnet_sparse import SparseCellNet

class ChargeSparseCellNet(torch.nn.Module):
    def __init__(self, nChannels, nHiddenLayers=1, freeze=False, **sparsekwargs):
        super().__init__()
        nOutputFeatures = 1
        self.sparseModel = SparseCellNet(nChannels=nChannels, **sparsekwargs)
        if freeze:
            for param in self.sparseModel.parameters():
                param.requires_grad = False
        linear_layers = []
        for _ in range(nHiddenLayers):
            linear_layers.append(nn.Linear(nChannels, nChannels))
            linear_layers.append(nn.ReLU())
        linear_layers.append(nn.Linear(nChannels, nOutputFeatures))
        self.linear = nn.Sequential(*linear_layers)

    def forward(self, point_cloud):
        """
        point_cloud is a list of length batch size = 1
        point_cloud[0] has 3 coordinates + batch index + features
        """
        point_cloud = point_cloud.squeeze(0)
        coords = point_cloud[:, :4].long()
        features = point_cloud[:, 4:].float()
        x = self.sparseModel((coords, features))
        x = self.linear(x)
        return x.unsqueeze(0)

def build_model(**kwargs):
    return ChargeSparseCellNet(**kwargs)
