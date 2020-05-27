import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel
import math
import torch
import numpy as np
import utils
from torch.autograd import Variable

####################### FSRCNN ##########################################

class FSRCNN(BaseModel):
    def __init__(self, scale_factor = 2, num_channels=3, d=56, s=12, m=4):
        super(FSRCNN, self).__init__()
        self.first_part = nn.Sequential(
            nn.Conv2d(num_channels, d, kernel_size=5, padding=5//2),
            nn.PReLU(d)
        )
        self.mid_part = [nn.Conv2d(d, s, kernel_size=1), nn.PReLU(s)]
        for _ in range(m):
            self.mid_part.extend([nn.Conv2d(s, s, kernel_size=3, padding=3//2), nn.PReLU(s)])
        self.mid_part.extend([nn.Conv2d(s, d, kernel_size=1), nn.PReLU(d)])
        self.mid_part = nn.Sequential(*self.mid_part)
        self.last_part = nn.ConvTranspose2d(d, num_channels, kernel_size=9, stride=scale_factor, padding=9//2,
                                            output_padding=scale_factor-1)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.first_part:
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight.data, mean=0.0, std=math.sqrt(2/(m.out_channels*m.weight.data[0][0].numel())))
                nn.init.zeros_(m.bias.data)
        for m in self.mid_part:
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight.data, mean=0.0, std=math.sqrt(2/(m.out_channels*m.weight.data[0][0].numel())))
                nn.init.zeros_(m.bias.data)
        nn.init.normal_(self.last_part.weight.data, mean=0.0, std=0.001)
        nn.init.zeros_(self.last_part.bias.data)

    def forward(self, x):
        x = self.first_part(x)
        x = self.mid_part(x)
        x = self.last_part(x)
        return x
    
    
############################## DRCN ##################################################

class DRCN(torch.nn.Module):
    def __init__(self, num_channels = 3, base_channel = 256, num_recursions = 16):
        super(DRCN, self).__init__()
        self.num_recursions = num_recursions
        self.embedding_layer = nn.Sequential(
            nn.Conv2d(num_channels, base_channel,
                      kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channel, base_channel,
                      kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )

        self.conv_block = nn.Sequential(nn.Conv2d(base_channel, base_channel, kernel_size=3, stride=1, padding=1),
                                        nn.ReLU(inplace=True))

        self.reconstruction_layer = nn.Sequential(
            nn.Conv2d(base_channel, base_channel,
                      kernel_size=3, stride=1, padding=1),
            nn.Conv2d(base_channel, num_channels,
                      kernel_size=3, stride=1, padding=1)
        )

        self.w_init = torch.ones(self.num_recursions) / self.num_recursions
        self.w = self.w_init

    def forward(self, x):
        h0 = self.embedding_layer(x)

        h = [h0]
        for d in range(self.num_recursions):
            h.append(self.conv_block(h[d]))

        y_d_ = list()
        out_sum = 0
        for d in range(self.num_recursions):
            y_d_.append(self.reconstruction_layer(h[d+1]))
            out_sum += torch.mul(y_d_[d], self.w[d])
        out_sum = torch.mul(out_sum, 1.0 / (torch.sum(self.w)))

        final_out = torch.add(out_sum, x)

        return y_d_, final_out

    def weight_init(self):
        for m in self._modules:
            weights_init_kaiming(m)


def weights_init_kaiming(m):
    class_name = m.__class__.__name__
    if class_name.find('Linear') != -1:
        torch.nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            m.bias.data.zero_()
    elif class_name.find('Conv2d') != -1:
        torch.nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            m.bias.data.zero_()
    elif class_name.find('ConvTranspose2d') != -1:
        torch.nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            m.bias.data.zero_()
    elif class_name.find('Norm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        if m.bias is not None:
            m.bias.data.zero_()
            
####################### LAPSRN #########################################


def upsample_filt(size):

    factor = (size + 1) // 2
    if size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:size, :size]
    return (1 - abs(og[0] - center) / factor) * \
           (1 - abs(og[1] - center) / factor)


def bilinear_upsample_weights(filter_size, weights):
    """
    Create weights matrix for transposed convolution with bilinear filter
    initialization.
    """
    f_out = weights.size(0)
    f_in = weights.size(1)
    weights = np.zeros((f_out,
                        f_in,
                        4,
                        4), dtype=np.float32)

    upsample_kernel = upsample_filt(filter_size)

    for i in range(f_out):
        for j in range(f_in):
            weights[i, j, :, :] = upsample_kernel
    return torch.Tensor(weights)


CNUM = 64  # 64


class FeatureExtraction(nn.Module):
    def __init__(self, level):
        super(FeatureExtraction, self).__init__()
        if level == 1:
            self.conv0 = nn.Conv2d(1, CNUM / 4, (3, 3), (1, 1), (1, 1))  # RGB
        else:
            self.conv0 = nn.Conv2d(CNUM, CNUM / 4, (3, 3), (1, 1), (1, 1))
        self.conv1 = nn.Conv2d(CNUM / 4, CNUM / 4, (3, 3), (1, 1), (1, 1))
        self.conv2 = nn.Conv2d(CNUM / 4, CNUM / 2, (3, 3), (1, 1), (1, 1))
        self.conv3 = nn.Conv2d(CNUM / 2, CNUM / 2, (3, 3), (1, 1), (1, 1))
        self.conv4 = nn.Conv2d(CNUM / 2, CNUM / 2, (3, 3), (1, 1), (1, 1))
        self.conv5 = nn.Conv2d(CNUM / 2, CNUM / 2, (3, 3), (1, 1), (1, 1))
        self.conv6 = nn.Conv2d(CNUM / 2, CNUM, (3, 3), (1, 1), (1, 1))
        self.conv7 = nn.Conv2d(CNUM, CNUM, (3, 3), (1, 1), (1, 1))
        self.conv8 = nn.Conv2d(CNUM, CNUM, (3, 3), (1, 1), (1, 1))
        self.conv9 = nn.Conv2d(CNUM, CNUM, (3, 3), (1, 1), (1, 1))
        self.conv10 = nn.Conv2d(CNUM, CNUM, (3, 3), (1, 1), (1, 1))
        self.convt_F = nn.ConvTranspose2d(CNUM, CNUM, (4, 4), (2, 2), (1, 1))
        self.LReLus = nn.LeakyReLU(negative_slope=0.1)
        self.convt_F.weight.data.copy_(
            bilinear_upsample_weights(4, self.convt_F.weight))

    def forward(self, x):
        out = self.LReLus(self.conv0(x))
        out = self.LReLus(self.conv1(out))
        out = self.LReLus(self.conv2(out))
        out = self.LReLus(self.conv3(out))
        out = self.LReLus(self.conv4(out))
        out = self.LReLus(self.conv5(out))
        out = self.LReLus(self.conv6(out))
        out = self.LReLus(self.conv7(out))
        out = self.LReLus(self.conv8(out))
        out = self.LReLus(self.conv9(out))
        out = self.LReLus(self.conv10(out))
        out = self.LReLus(self.convt_F(out))
        return out


class ImageReconstruction(nn.Module):
    def __init__(self):
        super(ImageReconstruction, self).__init__()
        self.conv_R = nn.Conv2d(CNUM, 1, (3, 3), (1, 1), (1, 1))  # RGB
        self.convt_I = nn.ConvTranspose2d(1, 1, (4, 4), (2, 2), (1, 1))  # RGB
        self.convt_I.weight.data.copy_(
            bilinear_upsample_weights(4, self.convt_I.weight))

    def forward(self, LR, convt_F):
        convt_I = self.convt_I(LR)
        conv_R = self.conv_R(convt_F)

        HR = convt_I + conv_R
        return HR


class LapSRN(nn.Module):
    def __init__(self):
        super(LapSRN, self).__init__()
        self.FeatureExtraction1 = FeatureExtraction(level=1)
        self.FeatureExtraction2 = FeatureExtraction(level=2)
        self.FeatureExtraction3 = FeatureExtraction(level=3)
        self.ImageReconstruction1 = ImageReconstruction()
        self.ImageReconstruction2 = ImageReconstruction()
        self.ImageReconstruction3 = ImageReconstruction()

    def forward(self, LR):
        convt_F1 = self.FeatureExtraction1(LR)
        HR_2 = self.ImageReconstruction1(LR, convt_F1)

        convt_F2 = self.FeatureExtraction2(convt_F1)
        HR_4 = self.ImageReconstruction2(HR_2, convt_F2)

        convt_F3 = self.FeatureExtraction3(convt_F2)
        HR_8 = self.ImageReconstruction3(HR_4, convt_F3)

        return HR_2, HR_4, HR_8



############################ DRRN ############################################

class DRRN(nn.Module):

    def __init__(self):
        super(DRRN, self).__init__()
        self.input = nn.Conv2d(in_channels=3, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv1 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv2 = nn.Conv2d(in_channels=128, out_channels=128,
		                       kernel_size=3, stride=1, padding=1, bias=False)
        self.output = nn.Conv2d(in_channels=128, out_channels=3, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)

		# weights initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))

    def forward(self, x):
        residual = x
        inputs = self.input(self.relu(x))
        out = inputs
        for _ in range(25):
            out = self.conv2(self.relu(self.conv1(self.relu(out))))
            out = torch.add(out, inputs)

        out = self.output(self.relu(out))
        out = torch.add(out, residual)
        return out

########################### MEMNET #######################################

class MemNet(nn.Module):
    def __init__(self, in_channels = 3, channels = 64, num_memblock = 6, num_resblock = 6):
        super(MemNet, self).__init__()
        self.feature_extractor = BNReLUConv(in_channels, channels)
        self.reconstructor = BNReLUConv(channels, in_channels)
        self.dense_memory = nn.ModuleList(
            [MemoryBlock(channels, num_resblock, i+1)
             for i in range(num_memblock)]
        )

    def forward(self, x):
        # x = x.contiguous()
        residual = x
        out = self.feature_extractor(x)
        ys = [out]
        for memory_block in self.dense_memory:
            out = memory_block(out, ys)
        out = self.reconstructor(out)
        out = out + residual

        return out


class MemoryBlock(nn.Module):
    """Note: num_memblock denotes the number of MemoryBlock currently"""

    def __init__(self, channels, num_resblock, num_memblock):
        super(MemoryBlock, self).__init__()
        self.recursive_unit = nn.ModuleList(
            [ResidualBlock(channels) for i in range(num_resblock)]
        )
        self.gate_unit = BNReLUConv(
            (num_resblock+num_memblock) * channels, channels, 1, 1, 0)

    def forward(self, x, ys):
        """ys is a list which contains long-term memory coming from previous memory block
        xs denotes the short-term memory coming from recursive unit
        """
        xs = []
        residual = x
        for layer in self.recursive_unit:
            x = layer(x)
            xs.append(x)

        gate_out = self.gate_unit(torch.cat(xs+ys, 1))
        ys.append(gate_out)
        return gate_out


class ResidualBlock(torch.nn.Module):
    """ResidualBlock
    introduced in: https://arxiv.org/abs/1512.03385
    x - Relu - Conv - Relu - Conv - x
    """

    def __init__(self, channels, k=3, s=1, p=1):
        super(ResidualBlock, self).__init__()
        self.relu_conv1 = BNReLUConv(channels, channels, k, s, p)
        self.relu_conv2 = BNReLUConv(channels, channels, k, s, p)

    def forward(self, x):
        residual = x
        out = self.relu_conv1(x)
        out = self.relu_conv2(out)
        out = out + residual
        return out


class BNReLUConv(nn.Sequential):
    def __init__(self, in_channels, channels, k=3, s=1, p=1, inplace=True):
        super(BNReLUConv, self).__init__()
        self.add_module('bn', nn.BatchNorm2d(in_channels))
        self.add_module('relu', nn.ReLU(inplace=inplace))
        self.add_module('conv', nn.Conv2d(
            in_channels, channels, k, s, p, bias=False))
