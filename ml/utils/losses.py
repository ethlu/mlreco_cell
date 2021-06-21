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

def DownsampleMSELoss(downsample, weight=1):
    import numpy as np
    def voxelize(pts):
        voxels = dict()
        for pt in pts:
            vox_i = tuple(pt[:3]//downsample)
            val = pt[3]
            if vox_i in voxels:
                voxels[vox_i] += val
            else:
                voxels[vox_i] = val
        return voxels
    def Loss(x, y):
        Y, f = y
        x, Y, f = x.squeeze(0), Y.squeeze(0), f[0]
        val_downsample, map_downsample = [], []
        with np.load(f, allow_pickle=True) as fi:
            event_starts = fi["starts"]
            pix_batch = fi["X"]
            Y_truth = fi["Y_truth"]
        for event in range(len(Y_truth)):
            event_starti = event_starts[event]
            if event == len(event_starts)-1:
                event_coords = pix_batch[event_starti:, :3]
            else:
                event_coords = pix_batch[event_starti:event_starts[event+1], :3]
            event_coords //= downsample
            truth_dict = voxelize(Y_truth[event])
            truth_dict_index = {}
            for coord in event_coords:
                coord = tuple(coord)
                if coord in truth_dict and coord not in truth_dict_index:
                    truth_dict_index[coord] = len(val_downsample)
                    val_downsample.append(truth_dict[coord])
            for coord in event_coords:
                coord = tuple(coord)
                if coord in truth_dict_index:
                    map_downsample.append(truth_dict_index[coord])
                else:
                    map_downsample.append(-1)
        val_downsample = torch.tensor(val_downsample)
        for i, val in zip(map_downsample, x.reshape(-1)):
            if i != -1:
                val_downsample[i] -= val

        single_mask = (Y>0).reshape(-1) | (torch.tensor(map_downsample)==-1)
        MaskedMSE = torch.mean((x[single_mask] - Y[single_mask])**2)
        DownsampleMSE = torch.mean(val_downsample**2)
        return MaskedMSE + weight*DownsampleMSE
    return Loss

def get_loss(loss_name):
    return globals()[loss_name]

if __name__=="__main__":
    print(get_loss("WeightedBCELoss")()(torch.ones((1,15,1))*0.5,
    torch.cat((torch.ones(1,5,1)*2,torch.ones(1,10,1)), 1)))
