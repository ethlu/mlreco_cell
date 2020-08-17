import torch

def BCEL1Loss(L1_lambda):
    BCE = torch.nn.BCELoss()
    L1 = torch.nn.L1Loss()
    def Loss(x, y):
        return BCE(x, y) + L1_lambda*L1(x, torch.zeros(x.shape))
    return Loss

def get_loss(loss_name):
    return globals()[loss_name]
