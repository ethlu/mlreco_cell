import torch
import sparseconvnet as scn

class BinarySparseCellNet(torch.nn.Module):
    def __init__(self, spatial_size, nChannels, nStrides, n_2D, reps=2, downsample=[2, 2], downsample_t=[4, 4]):
        super().__init__()
        dimension = 3
        nInputFeatures = 3
        nOutputFeatures = 1
        nPlanes = [i*nChannels for i in range(1, nStrides+1)]  # UNet number of features per level
        self.sparseModel = scn.Sequential().add(
           scn.InputLayer(dimension, spatial_size, mode=3)).add(
           scn.SubmanifoldConvolution(# Kernel size 3, no bias
               dimension, 
               nInputFeatures, 
               nChannels, 
               [1, 3, 3] if n_2D else 3,
               False)).add( 
           UResNet(nPlanes, n_2D, reps, downsample, downsample_t)).add(  # downsample = [filter size, filter stride]
           scn.BatchNormReLU(nChannels)).add( 
           scn.OutputLayer(dimension))
        self.linear = torch.nn.Linear(nChannels, nOutputFeatures)
        self.sigmoid = torch.nn.Sigmoid()

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
        x = self.sigmoid(x).double()
        return x.unsqueeze(0)

def UResNet(nPlanes, n_2D, reps, downsample=[2,2], downsample_t=[4,4], leakiness=0, n_input_planes=-1):
    dimension = 3
    def block(m, a, b, is2D): #ResNet style blocks
        kernel_size = [1, 3, 3] if is2D else 3 
        m.add(scn.ConcatTable()
              .add(scn.Identity() if a == b else scn.NetworkInNetwork(a, b, False))
              .add(scn.Sequential()
                .add(scn.BatchNormLeakyReLU(a,leakiness=leakiness))
                .add(scn.SubmanifoldConvolution(dimension, a, b, kernel_size, False))
                .add(scn.BatchNormLeakyReLU(b,leakiness=leakiness))
                .add(scn.SubmanifoldConvolution(dimension, b, b, kernel_size, False)))
         ).add(scn.AddTable())
    def U(nPlanes, n_2D, n_input_planes=-1): #Recursive function
        is2D = n_2D > 0
        down_conv_kernel = [(1 if is2D else downsample_t[0])] + downsample[:1]*2
        down_conv_stride = [(1 if is2D else downsample_t[1])] + downsample[1:]*2
        m = scn.Sequential()
        for i in range(reps):
            block(m, n_input_planes if n_input_planes!=-1 else nPlanes[0], nPlanes[0], is2D)
            n_input_planes=-1
        if len(nPlanes) > 1:
            m.add(
                scn.ConcatTable().add(
                    scn.Identity()).add(
                    scn.Sequential().add(
                        scn.BatchNormLeakyReLU(nPlanes[0],leakiness=leakiness)).add(
                        scn.Convolution(dimension, nPlanes[0], nPlanes[1],
                            down_conv_kernel, down_conv_stride, False)).add(
                        U(nPlanes[1:], n_2D-1)).add(
                        scn.BatchNormLeakyReLU(nPlanes[1],leakiness=leakiness)).add(
                        scn.Deconvolution(dimension, nPlanes[1], nPlanes[0],
                            down_conv_kernel, down_conv_stride, False))))
            m.add(scn.JoinTable())
            for i in range(reps):
                block(m, nPlanes[0] * (2 if i == 0 else 1), nPlanes[0], is2D)
        return m
    m = U(nPlanes, n_2D, n_input_planes)
    return m

def build_model(**kwargs):
    return BinarySparseCellNet(**kwargs)

if __name__ == "__main__":
    net = SparseCellNet(128, 2, 3, 2)
    print(net(torch.tensor([[[1, 2, 3, 1, 1, 2, 3]]])))
