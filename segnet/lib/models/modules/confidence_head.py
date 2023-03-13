''' 
Modeling Aleatoric Uncertainty for Camouflaged Object Detection
'''
import torch
import torch.nn as nn
import torch.nn.init as init

from lib.models.tools.module_helper import ModuleHelper


class UNetConvBlock(nn.Module):
    def __init__(self, in_size, out_size, bn_type):
        super(UNetConvBlock, self).__init__()

        self.conv = nn.Conv2d(in_size, out_size, kernel_size=3, stride=1,
                              padding=1, padding_mode='replicate')
        self.conv2 = nn.Conv2d(out_size, out_size, kernel_size=3, stride=1,
                               padding=1, padding_mode='replicate')
        self.bn = ModuleHelper.BatchNorm2d(bn_type=bn_type)(out_size)
        self.bn2 = ModuleHelper.BatchNorm2d(bn_type=bn_type)(out_size)
        self.activation = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.dropout = nn.Dropout(p=0.5)

        init.kaiming_normal_(self.conv.weight)
        init.kaiming_normal_(self.conv2.weight)

    def forward(self, x):
        out = self.activation(self.bn(self.conv(x)))
        out = self.activation(self.bn2(self.conv2(out)))
        out = self.dropout(out)
        return out


class UNetUpBlock(nn.Module):
    def __init__(self, in_size, out_size, bn_type):
        super(UNetUpBlock, self).__init__()

        self.conv1 = nn.ConvTranspose2d(in_size, out_size, kernel_size=2, stride=2, padding=0)
        # self.conv1 = nn.Conv2d(in_size, out_size, kernel_size=1, stride=1, padding=0)
        self.conv2_1 = nn.Conv2d(in_size, out_size, kernel_size=3, stride=1, padding=1)
        self.conv2_2 = nn.Conv2d(out_size, out_size, kernel_size=3, stride=1, padding=1)
        self.bn1 = ModuleHelper.BatchNorm2d(bn_type=bn_type)(out_size)
        self.bn2_1 = ModuleHelper.BatchNorm2d(bn_type=bn_type)(out_size)
        self.bn2_2 = ModuleHelper.BatchNorm2d(bn_type=bn_type)(out_size)
        self.activation = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.dropout_1 = nn.Dropout(p=0.5)
        self.dropout_2 = nn.Dropout(p=0.5)
        # self.interpolate = F.interpolate(scale_factor=2, mode='bilinear', align_corners=True)

        init.kaiming_normal_(self.conv1.weight)
        init.kaiming_normal_(self.conv2_1.weight)
        init.kaiming_normal_(self.conv2_2.weight)

    def forward(self, x, u):  # bridge is the corresponding lower layer
        # _, _, w, h = u.shape()
        # x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        out = self.activation(self.bn1(self.conv1(x)))
        out = self.dropout_1(out)
        out = torch.cat([out, u], dim=1)
        out = self.activation(self.bn2_1(self.conv2_1(out)))
        out = self.activation(self.bn2_2(self.conv2_2(out)))
        out = self.dropout_2(out)

        return out


class ConfidenceHead(nn.Module):
    def __init__(self, configer):
        super(ConfidenceHead, self).__init__()
        self.configer = configer
        self.num_classes = self.configer.get('data', 'num_classes')

        ndf = self.configer.get('protoseg', 'ndf_dim')
        bn_type = self.configer.get('network', 'bn_type')

        self.activation = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        # in_channel = self.num_classes + 3 # 22
        in_channel = 4
        self.down_block_1 = UNetConvBlock(in_channel, ndf, bn_type)
        self.down_block_2 = UNetConvBlock(ndf, 2*ndf, bn_type)
        self.down_block_3 = UNetConvBlock(2*ndf, 4*ndf, bn_type)
        self.down_block_4 = UNetConvBlock(4*ndf, 8*ndf, bn_type)
        self.down_block_5 = UNetConvBlock(8*ndf, 16*ndf, bn_type)

        self.up_block_4 = UNetUpBlock(16*ndf, 8*ndf, bn_type)
        self.up_block_3 = UNetUpBlock(8*ndf, 4*ndf, bn_type)
        self.up_block_2 = UNetUpBlock(4*ndf, 2*ndf, bn_type)
        self.up_block_1 = UNetUpBlock(2*ndf, ndf, bn_type)

        self.conv1 = nn.Conv2d(ndf, ndf, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(2*ndf, 2*ndf, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(4*ndf, 4*ndf, kernel_size=3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(8*ndf, 8*ndf, kernel_size=3, stride=2, padding=1)

        self.final_layer = nn.Conv2d(ndf, 1, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        block_1 = self.down_block_1(x)  # ndf*256*256
        x = self.conv1(block_1)
        block_2 = self.down_block_2(x)
        x = self.conv2(block_2)
        block_3 = self.down_block_3(x)
        x = self.conv3(block_3)
        block_4 = self.down_block_4(x)
        x = self.conv4(block_4)
        block_5 = self.down_block_5(x)

        out = self.up_block_4(block_5, block_4)
        out = self.up_block_3(out, block_3)
        out = self.up_block_2(out, block_2)
        out = self.up_block_1(out, block_1)

        out = self.final_layer(out)  # [b 1 h w]
        out = out.squeeze(1)  # [b h w]

        out = torch.sigmoid(out) #! confidence
        #! here we assume the output of confidence head is confidence not uncertainty!
        out = 1 - out #! uncertainty

        return out
