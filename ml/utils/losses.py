import torch

def BCEL1Loss(L1_lambda):
    BCE = torch.nn.BCELoss()
    L1 = torch.nn.L1Loss()
    def Loss(x, y):
        return BCE(x, y) + L1_lambda*L1(x, torch.zeros(x.shape))
    return Loss

def ChargeWeightedBCELoss(negative_weight):
    BCE = torch.nn.BCELoss(reduction='none')
    def Loss(x, y):
        labels = torch.where(y>0, torch.ones(y.shape), torch.zeros(y.shape))
        weights = torch.where(y>0, y, torch.full(y.shape, negative_weight))
        return torch.mean(BCE(x, labels)*weights)
    return Loss

def LabelWeightedBCELoss(positive_weight):
    BCE = torch.nn.BCELoss(reduction='none')
    def Loss(x, y):
        weights = torch.where(y==1,
                torch.full(y.shape, positive_weight),
                torch.ones(y.shape))
        return torch.mean(BCE(x, y)*weights)
    return Loss

def get_loss(loss_name):
    return globals()[loss_name]

if __name__=="__main__":
    print(get_loss("WeightedBCELoss")()(torch.ones((1,15,1))*0.5,
    torch.cat((torch.ones(1,5,1)*2,torch.ones(1,10,1)), 1)))
