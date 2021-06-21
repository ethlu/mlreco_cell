import torch
from models.cellnet_sparse import SparseCellNet

class BinarySparseCellNet(torch.nn.Module):
    def __init__(self, nChannels, **sparsekwargs):
        super().__init__()
        nOutputFeatures = 1
        self.sparseModel = SparseCellNet(nChannels=nChannels, **sparsekwargs)
        self.linear = torch.nn.Linear(nChannels, nOutputFeatures)

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
        x = torch.sigmoid(x)
        return x.unsqueeze(0)

def build_model(**kwargs):
    return BinarySparseCellNet(**kwargs)

