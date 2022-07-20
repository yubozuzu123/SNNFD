import os
import sys
import numpy as np
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
import torch_dct as dct

pretrained_settings = {
    'xception': {
        'imagenet': {
            'url': 'http://data.lip6.fr/cadene/pretrainedmodels/xception-b5690688.pth',
            'input_space': 'RGB',
            'input_size': [3, 299, 299],
            'input_range': [0, 1],
            'mean': [0.5, 0.5, 0.5],
            'std': [0.5, 0.5, 0.5],
            'num_classes': 1000,
            'scale': 0.8975 # The resize parameter of the validation transform should be 333, and make sure to center crop at 299x299
        }
    }
}
class Upsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.PReLU()
        )

    def forward(self, x):
        h, w = 3 * x.size(2), 3 * x.size(3)
        p = F.upsample(input=x, size=(h, w), mode='bilinear')
        return self.conv(p)

class Upsample_final(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.PReLU()
        )

    def forward(self, x):
        p = F.upsample(input=x, size=(512, 512), mode='bilinear')
        return self.conv(p)
class F3Net(nn.Module):
    """
    Implementation is mainly referenced from https://github.com/yyk-wew/F3Net
    """
    def __init__(self, 
                 num_classes: int=2, 
                 img_width: int=512, 
                 img_height: int=512, 
                 LFS_window_size: int=11, 
                 LFS_M: int=3) -> None:
        super(F3Net, self).__init__()
        assert img_width == img_height
        self.img_size = img_width
        self.num_classes = num_classes
        self._LFS_window_size = LFS_window_size
        self._LFS_M = LFS_M
        
        
        self.fad_head = FAD_Head(self.img_size)
        self.lfs_head = LFS_Head(self.img_size, self._LFS_window_size, self._LFS_M)
        
        self.fad_excep = self._init_xcep_fad()
        self.lfs_excep = self._init_xcep_lfs()
        
        self.excep_forwards = ['conv1', 'bn1', 'relu', 'conv2', 'bn2', 'relu', 
                               'block1', 'block2', 'block3', 'block4', 'block5', 'block6','block7', 
                               'block8', 'block9', 'block10' , 'block11', 'block12']

         # classifier
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(4096, num_classes)
        self.dp = nn.Dropout(p=0.2)
        self.upsample1=Upsample(1024, 512)
        self.upsample2=Upsample(512, 256)
        self.upsample3=Upsample_final(256, 64)
        self.upsample4=Upsample_final(64, 12)
        self.final = nn.Sequential(
            nn.Conv2d(64, num_classes, kernel_size=1),
            nn.LogSoftmax()
        )

        self.sample2=nn.Conv2d(64, 12, kernel_size=1)
        self.sample=nn.Conv2d(363, 64, kernel_size=1)
        self.conv0=nn.Conv2d(24, 3, kernel_size=1)
        self.conv01=nn.Conv2d(6, 3, kernel_size=1)
        self.xception=Xception()
    def _init_xcep_fad(self):
        fad_excep =  return_pytorch04_xception(True)
        #conv1_data = fad_excep.conv1.weight.data
        # let new conv1 use old param to balance the network
        #fad_excep.conv1 = nn.Conv2d(24, 32, 3, 2, 0, bias=False)
        #for i in range(8):
        #    fad_excep.conv1.weight.data[:, i*3:(i+1)*3, :, :] = conv1_data /8.0
        return fad_excep
    
    def  _init_xcep_lfs(self): 
        lfs_excep = return_pytorch04_xception(True)
        conv1_data = lfs_excep.conv1.weight.data
        # let new conv1 use old param to balance the network
        lfs_excep.conv1 = nn.Conv2d(self._LFS_M, 32, 3, 1, 0, bias=False)
        for i in range(int(self._LFS_M / 3)):
            lfs_excep.conv1.weight.data[:, i*3:(i+1)*3, :, :] = conv1_data / float(self._LFS_M / 3.0)
        return lfs_excep
    
    def _features(self, x_fad):
        for forward_func in self.excep_forwards:
            x_fad = getattr(self.fad_excep, forward_func)(x_fad)
        return x_fad
    
    def _norm_feature(self, x):
        x = self.relu(x)
        #print(x.shape)
        x = F.adaptive_avg_pool2d(x, (1,1))
        #print(x.shape)
        x = x.view(x.size(0), -1)
        return x
    
    def forward(self, x):
        fad_input = self.fad_head(x)
        lfs_input = self.lfs_head(x)
        lfs_input_compress= self.sample(lfs_input)
        lfs_input_compress = self.sample2(lfs_input_compress)
        #print(lfs_input_compress.shape)
        x_cat = torch.cat((fad_input, lfs_input_compress), dim=1)
        x_space_cat = []
        for i in range(24):
            x_tmp=x_cat[:,i,:,:]
            x_tmp=torch.squeeze(x_tmp)
            x_space_tmp = dct.idct(x_tmp)
            x_space_tmp =torch.unsqueeze(x_space_tmp,1) 
            x_space_cat.append(x_space_tmp)
        #finish idct and start to convolute in the spatial domain
        x_space_cat_cat = torch.cat(x_space_cat, dim=1) 
        x_feature_cat=self.conv0(x_space_cat_cat)
        x_space_cat_cat = torch.cat((x_feature_cat,x), dim=1) 
        x_feature = self.conv01(x_space_cat_cat)    
        x_feature = self._features(x_feature)                
        x_feature = self.upsample1(x_feature)
        x_feature = self.upsample2(x_feature)  
        x_feature = self.upsample3(x_feature)
        x_feature = self.final(x_feature)
        return x_feature

class SeparableConv2d(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size=1,stride=1,padding=0,dilation=1,bias=False):
        super(SeparableConv2d,self).__init__()

        self.conv1 = nn.Conv2d(in_channels,in_channels,kernel_size,stride,padding,dilation,groups=in_channels,bias=bias)
        self.pointwise = nn.Conv2d(in_channels,out_channels,1,1,0,1,1,bias=bias)

    def forward(self,x):
        x = self.conv1(x)
        x = self.pointwise(x)
        return x

class Block(nn.Module):
    def __init__(self,in_filters,out_filters,reps,strides=1,start_with_relu=True,grow_first=True):
        super(Block, self).__init__()

        if out_filters != in_filters or strides!=1:
            self.skip = nn.Conv2d(in_filters,out_filters,1,stride=strides, bias=False)
            self.skipbn = nn.BatchNorm2d(out_filters)
        else:
            self.skip=None

        self.relu = nn.ReLU(inplace=True)
        rep=[]

        filters=in_filters
        if grow_first:
            rep.append(self.relu)
            rep.append(SeparableConv2d(in_filters,out_filters,3,stride=1,padding=1,bias=False))
            rep.append(nn.BatchNorm2d(out_filters))
            filters = out_filters

        for i in range(reps-1):
            rep.append(self.relu)
            rep.append(SeparableConv2d(filters,filters,3,stride=1,padding=1,bias=False))
            rep.append(nn.BatchNorm2d(filters))

        if not grow_first:
            rep.append(self.relu)
            rep.append(SeparableConv2d(in_filters,out_filters,3,stride=1,padding=1,bias=False))
            rep.append(nn.BatchNorm2d(out_filters))

        if not start_with_relu:
            rep = rep[1:]
        else:
            rep[0] = nn.ReLU(inplace=False)

        if strides != 1:
            rep.append(nn.MaxPool2d(3,strides,1))
        self.rep = nn.Sequential(*rep)

    def forward(self,inp):
        x = self.rep(inp)

        if self.skip is not None:
            skip = self.skip(inp)
            skip = self.skipbn(skip)
        else:
            skip = inp

        x+=skip
        return x

class Xception(nn.Module):
    """
    Xception optimized for the ImageNet dataset, as specified in
    https://arxiv.org/pdf/1610.02357.pdf
    """
    def __init__(self, num_classes=1000):
        """ Constructor
        Args:
            num_classes: number of classes
        """
        super(Xception, self).__init__()
        self.num_classes = num_classes

        self.conv1 = nn.Conv2d(3, 32, 3,2, 0, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(32,64,3,bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        #do relu here

        self.block1=Block(64,128,2,2,start_with_relu=False,grow_first=True)
        self.block2=Block(128,256,2,2,start_with_relu=True,grow_first=True)
        self.block3=Block(256,728,2,2,start_with_relu=True,grow_first=True)

        self.block4=Block(728,728,3,1,start_with_relu=True,grow_first=True)
        self.block5=Block(728,728,3,1,start_with_relu=True,grow_first=True)
        self.block6=Block(728,728,3,1,start_with_relu=True,grow_first=True)
        self.block7=Block(728,728,3,1,start_with_relu=True,grow_first=True)

        self.block8=Block(728,728,3,1,start_with_relu=True,grow_first=True)
        self.block9=Block(728,728,3,1,start_with_relu=True,grow_first=True)
        self.block10=Block(728,728,3,1,start_with_relu=True,grow_first=True)
        self.block11=Block(728,728,3,1,start_with_relu=True,grow_first=True)

        self.block12=Block(728,1024,2,2,start_with_relu=True,grow_first=False)

        self.conv3 = SeparableConv2d(1024,1536,3,1,1)
        self.bn3 = nn.BatchNorm2d(1536)

        #do relu here
        self.conv4 = SeparableConv2d(1536,2048,3,1,1)
        self.bn4 = nn.BatchNorm2d(2048)
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)
        x = self.block8(x)
        x = self.block9(x)
        x = self.block10(x)
        x = self.block11(x)
        x = self.block12(x)
      
        #x = self.conv3(x)
        #x = self.bn3(x)
        #x = self.conv4(x)
        #x = self.bn4(x)
        #print(x.shape)
        return x

def xception(num_classes=1000, pretrained='imagenet'):
    model = Xception(num_classes=num_classes)
    if pretrained:
        settings = pretrained_settings['xception'][pretrained]
        assert num_classes == settings['num_classes'], \
            "num_classes should be {}, but is {}".format(settings['num_classes'], num_classes)

        model = Xception(num_classes=num_classes)
        model.load_state_dict(model_zoo.load_url(settings['url']))

        model.input_space = settings['input_space']
        model.input_size = settings['input_size']
        model.input_range = settings['input_range']
        model.mean = settings['mean']
        model.std = settings['std']

    return model

def return_pytorch04_xception(pretrained=True):
    model = xception(pretrained=False)
    if pretrained:
        state_dict = torch.load(
            'pretrained/xception-b5690688.pth')
        for name, weights in state_dict.items():
            if 'pointwise' in name:
                state_dict[name] = weights.unsqueeze(-1).unsqueeze(-1)
        model.load_state_dict(state_dict, strict=False)
 
    return model

class Filter(nn.Module):
    def __init__(self, size, 
                 band_start, 
                 band_end, 
                 use_learnable=True, 
                 norm=False):
        super(Filter, self).__init__()
        self.use_learnable = use_learnable

        self.base = nn.Parameter(torch.tensor(generate_filter(band_start, band_end, size)), requires_grad=False)
        if self.use_learnable:
            self.learnable = nn.Parameter(torch.randn(size, size), requires_grad=True)
            self.learnable.data.normal_(0., 0.1)
            # Todo
            # self.learnable = nn.Parameter(torch.rand((size, size)) * 0.2 - 0.1, requires_grad=True)

        self.norm = norm
        if norm:
            self.ft_num = nn.Parameter(torch.sum(torch.tensor(generate_filter(band_start, band_end, size))), requires_grad=False)


    def forward(self, x):
        if self.use_learnable:
            filt = self.base + norm_sigma(self.learnable)
        else:
            filt = self.base

        if self.norm:
            y = x * filt / self.ft_num
        else:
            y = x * filt
        return y

class FAD_Head(nn.Module):
    def __init__(self, size):
        super(FAD_Head, self).__init__()
        # init DCT matrix
        self._DCT_all = nn.Parameter(torch.tensor(DCT_mat(size)).float(), requires_grad=False)
        self._DCT_all_T = nn.Parameter(torch.transpose(torch.tensor(DCT_mat(size)).float(), 0, 1), requires_grad=False)

        # define base filters and learnable
        # 0 - 1/16 || 1/16 - 1/8 || 1/8 - 1
        low_filter = Filter(size, 0, size // 50)
        middle_filter = Filter(size, size // 50, size // 10)
        high_filter = Filter(size, size // 10, size)
        all_filter = Filter(size, 0, size * 2)

        self.filters = nn.ModuleList([low_filter, middle_filter, high_filter, all_filter])
        
        self.k1=nn.Parameter(torch.FloatTensor(1),requires_grad=True)
        self.k2=nn.Parameter(torch.FloatTensor(1),requires_grad=True)
        self.k3=nn.Parameter(torch.FloatTensor(1),requires_grad=True)
        self.k4=nn.Parameter(torch.FloatTensor(1),requires_grad=True)
        self.k1.data.fill_(0.25)
        self.k2.data.fill_(0.25)
        self.k3.data.fill_(0.25)
        self.k4.data.fill_(0.25)
    def forward(self, x):
        # DCT
        x_freq = self._DCT_all @ x @ self._DCT_all_T    # [N, 3, 299, 299]

        # 4 kernel
        y_list = []
        for i in range(4):
            x_pass = self.filters[i](x_freq)  # [N, 3, 299, 299]
            y = self._DCT_all_T @ x_pass @ self._DCT_all    # [N, 3, 299, 299]
            y_list.append(y)
        y_list[0]=y_list[0]*self.k1
        y_list[1]=y_list[1]*self.k2
        y_list[2]=y_list[2]*self.k3
        y_list[3]=y_list[3]*self.k4
        
        out = torch.cat(y_list, dim=1)    # [N, 12, 299, 299]
        
        return out


class LFS_Head(nn.Module):
    def __init__(self, size, window_size, M):
        super(LFS_Head, self).__init__()

        self.window_size = window_size
        self._M = M

        # init DCT matrix
        self._DCT_patch = nn.Parameter(torch.tensor(DCT_mat(window_size)).float(), requires_grad=False)
        self._DCT_patch_T = nn.Parameter(torch.transpose(torch.tensor(DCT_mat(window_size)).float(), 0, 1), requires_grad=False)

        #self.unfold = nn.Unfold(kernel_size=(window_size, window_size),stride=1, padding=1)
        self.unfold = nn.Unfold(kernel_size=(window_size, window_size), stride=1, padding=5)
        # init filters
        self.filters = nn.ModuleList([Filter(window_size, window_size * 2. / M * i, window_size * 2. / M * (i+1), norm=True) for i in range(M)])
        self.k1=nn.Parameter(torch.FloatTensor(1),requires_grad=True)
        self.k2=nn.Parameter(torch.FloatTensor(1),requires_grad=True)
        self.k3=nn.Parameter(torch.FloatTensor(1),requires_grad=True)
        self.k1.data.fill_(0.25)
        self.k2.data.fill_(0.25)
        self.k3.data.fill_(0.25)
    
    def forward(self, x):
        # turn RGB into Gray
        x_gray = 0.299*x[:,0,:,:] + 0.587*x[:,1,:,:] + 0.114*x[:,2,:,:]
        x = x_gray.unsqueeze(1)

        # rescale to 0 - 255
        x = (x + 1.) * 122.5

        # calculate size
        N, C, W, H = x.size()
        S = self.window_size
        size_after = int((W+2*5-(S-1)-1)/1) + 1
        
        

        # sliding window unfold and DCT
        x_unfold = self.unfold(x)   # [N, C * S * S, L]   L:block num       
        L = x_unfold.size()[2]
        x_unfold = x_unfold.transpose(1, 2).reshape(N, L, C, S, S)  # [N, L, C, S, S]
        x_dct = self._DCT_patch @ x_unfold @ self._DCT_patch_T     
        # M kernels filtering
        y_list = []
        for i in range(self._M):
            y = torch.abs(x_dct)
            #print(y.shape)
            y = torch.log10(y + 1e-15)
            y = self.filters[i](y)
            y  = y .reshape(N, size_after*size_after,S*S)
            y  = torch.transpose(y ,2,1)
            y  = y .reshape(N,S*S,size_after,size_after)  # [N, 1, 149, 149]
            y_list.append(y)
        y_list[0]=y_list[0]*self.k1
        y_list[1]=y_list[1]*self.k2
        y_list[2]=y_list[2]*self.k3
        out = torch.cat(y_list, dim=1)  # [N, M, 149, 149]
        return out
    
class MixBlock(nn.Module):
    
    def __init__(self, c_in, width, height):
        super(MixBlock, self).__init__()
        self.FAD_query = nn.Conv2d(c_in, c_in, (1,1))
        self.LFS_query = nn.Conv2d(c_in, c_in, (1,1))

        self.FAD_key = nn.Conv2d(c_in, c_in, (1,1))
        self.LFS_key = nn.Conv2d(c_in, c_in, (1,1))

        self.softmax = nn.Softmax(dim=-1)
        self.relu = nn.ReLU()

        self.FAD_gamma = nn.Parameter(torch.zeros(1))
        self.LFS_gamma = nn.Parameter(torch.zeros(1))

        self.FAD_conv = nn.Conv2d(c_in, c_in, (1,1), groups=c_in)
        self.FAD_bn = nn.BatchNorm2d(c_in)
        self.LFS_conv = nn.Conv2d(c_in, c_in, (1,1), groups=c_in)
        self.LFS_bn = nn.BatchNorm2d(c_in)

    def forward(self, x_FAD, x_LFS):
        B, C, W, H = x_FAD.size()
        assert W == H

        q_FAD = self.FAD_query(x_FAD).view(-1, W, H)    # [BC, W, H]
        q_LFS = self.LFS_query(x_LFS).view(-1, W, H)
        M_query = torch.cat([q_FAD, q_LFS], dim=2)  # [BC, W, 2H]

        k_FAD = self.FAD_key(x_FAD).view(-1, W, H).transpose(1, 2)  # [BC, H, W]
        k_LFS = self.LFS_key(x_LFS).view(-1, W, H).transpose(1, 2)
        M_key = torch.cat([k_FAD, k_LFS], dim=1)    # [BC, 2H, W]

        energy = torch.bmm(M_query, M_key)  #[BC, W, W]
        attention = self.softmax(energy).view(B, C, W, W)

        att_LFS = x_LFS * attention * (torch.sigmoid(self.LFS_gamma) * 2.0 - 1.0)
        y_FAD = x_FAD + self.FAD_bn(self.FAD_conv(att_LFS))

        att_FAD = x_FAD * attention * (torch.sigmoid(self.FAD_gamma) * 2.0 - 1.0)
        y_LFS = x_LFS + self.LFS_bn(self.LFS_conv(att_FAD))
        return y_FAD, y_LFS

# utils
def DCT_mat(size):
    m = [[ (np.sqrt(1./size) if i == 0 else np.sqrt(2./size)) * np.cos((j + 0.5) * np.pi * i / size) for j in range(size)] for i in range(size)]
    return m

def generate_filter(start, end, size):
    return [[0. if i + j > end or i + j <= start else 1. for j in range(size)] for i in range(size)]

def norm_sigma(x):
    return 2. * torch.sigmoid(x) - 1.
    

   
