import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from my_config import config

H=config['Height']
W=config['Width']
D=config['M_D']
BS=1
Feature=config['Feature_size']

class ResidualBlock(nn.Module):
    def __init__(self):
        super(ResidualBlock,self).__init__()
        self.conv1=nn.Conv2d(32,32,3,padding=1)
        self.bn1=nn.BatchNorm2d(32)
        self.conv2=nn.Conv2d(32,32,3,padding=1)
        self.bn2=nn.BatchNorm2d(32)

    def forward(self):
        x_stored=x
        x=self.conv1(x)
        x=self.bn1(x)
        x=F.relu(x)
        x=self.conv2(x)
        x=self.bn2(x)
        x=F.relu(x)
        x=x+x_stored

        return x

class UnaryFeatures(nn.Module):
    def __init__(self):
        super(UnaryFeatures,self).__init__()
        self.seq=nn.Sequential(
            nn.Conv2d(3,32,5,2,2),
            ResidualBlock(),
            ResidualBlock(),
            ResidualBlock(),
            ResidualBlock(),
            ResidualBlock(),
            ResidualBlock(),
            ResidualBlock(),
            ResidualBlock(),
            nn.Conv2d(32,32,3,1,1)
        )
    def forward(self, x):
        x=self.seq(x)
        return x

class SubSample3D(nn.Module):
    def __init__(self,inchannels, outchannels):
        super(SubSample3D,self).__init__()

        self.conv1=nn.Conv3d(inchannels,outchannels,3,1,1)
        self.bn1=nn.BatchNorm3d(outchannels)

        self.conv2=nn.Conv3d(outchannels,outchannels,3,1,1)
        self.bn2=nn.BatchNorm3d(outchannels)

    def forward(self, x):
        x=self.conv1(x)
        x=self.bn1(x)
        x=F.relu(x)

        x=self.conv2(x)
        x=self.bn2(x)
        x=F.relu(x)

        return x

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder,self).__init__()
        self.sub1_sample=SubSample3D(64,32)
        self.sub1_conv1=nn.Conv3d(64,32,3,1,1)
        self.sub1_bn_1=nn.BatchNorm3d(32)
        self.sub1_conv2=nn.Conv3d(32,32,3,1,1)
        self.sub1_bn_2=nn.BatchNorm3d(32)

        self.conv1=nn.Conv3d(64,64,3,2,1)
        self.bn1=nn.BatchNorm3d(64)
        self.subsample1=SubSample3D(64,64)

        self.conv2=nn.Conv3d(64,64,3,2,1)
        self.bn2=nn.BatchNorm3d(64)
        self.subsample2=SubSample3D(64,64)

        self.conv3=nn.Conv3d(64,64,3,2,1)
        self.bn3=nn.BatchNorm3d(64)
        self.subsample3=SubSample3D(64,64)

        self.conv4=nn.Conv3d(64,128,3,2,1)
        self.bn4=nn.BatchNorm3d(128)
        self.subsample4=SubSample3D(128,128)

    def forward(self,x):
        x=self.conv1(x)
        x=self.bn1(x)
        x=F.relu(x)

        x=self.conv2(x)
        x = self.bn2(x)
        x=F.relu(x)

        x=self.conv3(x)
        x=self.bn3(x)
        x=F.relu(x)

        x=self.conv4(x)
        x=self.bn4(x)
        x=F.relu(x)
        x=self.subsample4(x)

        return x

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder,self).__init__()
        self.conv_trans3d1=nn.ConvTranspose3d(128,64,3,2,1,(1,1,1))
        self.bn1=nn.BatchNorm3d(64)

        self.conv_trans3d2=nn.ConvTranspose3d(64,64,3,2,1,(1,1,1))
        self.bn2=nn.BatchNorm3d(64)

        self.conv_trans3d3=nn.ConvTranspose3d(64,64,3,2,1,(1,1,1))
        self.bn3=nn.BatchNorm3d(32)

        self.conv_trans3d4=nn.ConvTranspose3d(64,32,3,2,1,(1,1,1))
        self.bn3=nn.BatchNorm3d(32)

        self.conv_trans3d5=nn.ConvTranspose3d(32,1,3,2,1,(1,1,1))

    def forward(self, x):
        x=self.conv_trans3d1(x)
        x=self.bn1(x)
        x=F.relu(x)

        x=self.conv_trans3d2(x)
        x=self.bn2(x)
        x=F.relu(x)

        x=self.conv_trans3d3(x)
        x=self.bn3(x)
        x=F.relu(x)

        x=self.conv_trans3d4(x)
        x=self.bn4(x)
        x=F.relu(x)

        x=self.conv_trans3d4(x)

        return x

class GCNet(nn.Module):
    def __init__(self):
        super(GCNet,self).__init__()

        self.unaryfeatures=UnaryFeatures()
        self.encoder=Encoder()
        self.decoder=Decoder()

    def forward(self, x):
        left_image,right_image=x

        left_features=self.unaryfeatures(left_image)
        right_feature=self.unaryfeatures(right_image)

        cost_volume=self._get_cost_volume(left_features,right_feature)

        cost_volume=self.encoder(cost_volume)

        cost_volume=self.decoder(cost_volume)

        disp_preditction=self.soft_argmin(cost_volume)

        return disp_preditction

    def _get_cost_volume(self, left_feature, right_feature):
        d=int(D/2)+1
        [bs,f,h,w]=left_feature.size()
        # cost_volume=Variable(torch.zeros((bs,2*f,d-1,h,w)).cuda())
        cost_volume = Variable ( torch.zeros ( (bs, 2 * f, d - 1, h, w) ))
        for d_i in range(1,d):
            cost_volume[:,0:f,d_i-1,:,:]=left_feature
            cost_volume[:,f:2*f,d_i-1,:,d_i:w]=right_feature[:,:,:,0:w-d_i]
        return cost_volume

    def soft_argmin(self, cost_volume):
        pro_volume=-cost_volume
        pro_volume=torch.squeeze(pro_volume,dim=0)
        pro_volume=torch.squeeze(pro_volume,dim=0)

        pro_volume=F.softmax(pro_volume)
        d=pro_volume.size()[0]
        # pro_volume_indices=Variable(torch.zeros(pro_volume.size()).cuda())
        pro_volume_indices = Variable ( torch.zeros ( pro_volume.size () ))
        for i in range(d):
            pro_volume_indices[i]=(i+1)*pro_volume[i]
        disp_prediction=torch.sum(pro_volume_indices,dim=0)
        return disp_prediction

if __name__=='__main__':
    print('... debug net work!')
    net = GCNet ()
    # print(len(list(net.parameters())))
    # print(len(list(net.unaryfeatures.parameters())))
    # print(len(list(net.encoder.parameters())))
    # print(len(list(net.decoder.parameters())))
    # sub=SubSample3D(32,64)
    # print(len(list(sub.parameters())))
    # print('...debug net work!')
    # net = GCNet ().cuda ()
    # x1 = torch.rand ( (1, 3, 256, 512) ).cuda ()
    # x2 = torch.rand ( (1, 3, 256, 512) ).cuda ()
    x1 = torch.rand ( (1, 3, 256, 512) )
    x2 = torch.rand ( (1, 3, 256, 512) )
    x = (Variable ( x1 ), Variable ( x2 ))
    y = net ( x )
    # time.sleep ( 100 )
    print(y)




