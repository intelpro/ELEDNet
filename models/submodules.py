import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import init
from torch.nn.modules.batchnorm import _BatchNorm


def actFunc(act, *args, **kwargs):
    act = act.lower()
    if act == 'relu':
        return nn.ReLU()
    elif act == 'relu6':
        return nn.ReLU6()
    elif act == 'leakyrelu':
        return nn.LeakyReLU(0.1)
    elif act == 'prelu':
        return nn.PReLU()
    elif act == 'rrelu':
        return nn.RReLU(0.1, 0.3)
    elif act == 'selu':
        return nn.SELU()
    elif act == 'celu':
        return nn.CELU()
    elif act == 'elu':
        return nn.ELU()
    elif act == 'gelu':
        return nn.GELU()
    elif act == 'tanh':
        return nn.Tanh()
    else:
        raise NotImplementedError

class ResBlock(nn.Module):
    """
    Residual block
    """
    def __init__(self, in_chs, activation='relu', batch_norm=False):
        super(ResBlock, self).__init__()
        op = []
        for i in range(2):
            op.append(conv3x3(in_chs, in_chs))
            if batch_norm:
                op.append(nn.BatchNorm2d(in_chs))
            if i == 0:
               op.append(actFunc(activation))
        self.main_branch = nn.Sequential(*op)

    def forward(self, x):
        out = self.main_branch(x)
        out += x
        return out


### resblock
def make_layer(basic_block, num_basic_block, **kwarg):
    """Make layers by stacking the same blocks.

    Args:
        basic_block (nn.module): nn.module class for basic block.
        num_basic_block (int): number of blocks.

    Returns:
        nn.Sequential: Stacked blocks in nn.Sequential.
    """
    layers = []
    for _ in range(num_basic_block):
        layers.append(basic_block(**kwarg))
    return nn.Sequential(*layers)

@torch.no_grad()
def default_init_weights(module_list, scale=1, bias_fill=0, **kwargs):
    """Initialize network weights.

    Args:
        module_list (list[nn.Module] | nn.Module): Modules to be initialized.
        scale (float): Scale initialized weights, especially for residual
            blocks. Default: 1.
        bias_fill (float): The value to fill bias. Default: 0
        kwargs (dict): Other arguments for initialization function.
    """
    if not isinstance(module_list, list):
        module_list = [module_list]
    for module in module_list:
        for m in module.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, **kwargs)
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, **kwargs)
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)
            elif isinstance(m, _BatchNorm):
                init.constant_(m.weight, 1)
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)


class ResidualBlockNoBN3D(nn.Module):
    """Residual block without BN.

    It has a style of:
        ---Conv-ReLU-Conv-+-
         |________________|

    Args:
        num_feat (int): Channel number of intermediate features.
            Default: 64.
        res_scale (float): Residual scale. Default: 1.
        pytorch_init (bool): If set to True, use pytorch default init,
            otherwise, use default_init_weights. Default: False.
    """

    def __init__(self, num_feat=64, res_scale=1, pytorch_init=False):
        super(ResidualBlockNoBN3D, self).__init__()
        self.res_scale = res_scale
        self.conv1 = nn.Conv3d(num_feat, num_feat, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv3d(num_feat, num_feat, 3, 1, 1, bias=True)
        self.relu = nn.ReLU(inplace=True)

        if not pytorch_init:
            default_init_weights([self.conv1, self.conv2], 0.1)

    def forward(self, x):
        identity = x
        out = self.conv2(self.relu(self.conv1(x)))
        return identity + out * self.res_scale

class ResidualBlockNoBN(nn.Module):
    """Residual block without BN.

    It has a style of:
        ---Conv-ReLU-Conv-+-
         |________________|

    Args:
        num_feat (int): Channel number of intermediate features.
            Default: 64.
        res_scale (float): Residual scale. Default: 1.
        pytorch_init (bool): If set to True, use pytorch default init,
            otherwise, use default_init_weights. Default: False.
    """

    def __init__(self, num_feat=64, pytorch_init=False):
        super(ResidualBlockNoBN, self).__init__()
        self.conv1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=True)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        if not pytorch_init:
            default_init_weights([self.conv1, self.conv2], 0.1)

    def forward(self, x):
        identity = x
        out = self.conv2(self.lrelu(self.conv1(x)))
        return identity + out
    

class ResidualBlockNoBN2D(nn.Module):
    """Residual block without BN.

    It has a style of:
        ---Conv-ReLU-Conv-+-
         |________________|

    Args:
        num_feat (int): Channel number of intermediate features.
            Default: 64.
        res_scale (float): Residual scale. Default: 1.
        pytorch_init (bool): If set to True, use pytorch default init,
            otherwise, use default_init_weights. Default: False.
    """

    def __init__(self, num_feat=64, res_scale=1, pytorch_init=False):
        super(ResidualBlockNoBN2D, self).__init__()
        self.res_scale = res_scale
        self.conv1 = nn.Conv3d(num_feat, num_feat, (1, 3, 3), 1, (0, 1, 1), bias=True)
        self.conv2 = nn.Conv3d(num_feat, num_feat, (1, 3, 3), 1, (0, 1, 1), bias=True)
        self.relu = nn.ReLU(inplace=True)

        if not pytorch_init:
            default_init_weights([self.conv1, self.conv2], 0.1)

    def forward(self, x):
        identity = x
        out = self.conv2(self.relu(self.conv1(x)))
        return identity + out * self.res_scale

class ResidualBlocks2D(nn.Module):
    def __init__(self, num_feat=64, num_block=30):
        super().__init__()
        self.main = nn.Sequential(
            make_layer(ResidualBlockNoBN, num_block, num_feat=num_feat))

    def forward(self, fea):
        return self.main(fea)

# conv blocks 
def conv1x1(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0, bias=True)

def conv1x1_relu(in_channels, out_channels, stride=1):
    return nn.Sequential(conv1x1(in_channels, out_channels, stride), nn.ReLU())

def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=True)

def conv3x3_relu(in_channels, out_channels, stride=1):
    return nn.Sequential(conv3x3(in_channels, out_channels, stride), nn.ReLU())

def conv5x5(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=5, stride=stride, padding=2, bias=True)

def deconv4x4(in_channels, out_channels, stride=2):
    return nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=stride, padding=1)

def deconv5x5(in_channels, out_channels, stride=2):
    return nn.ConvTranspose2d(in_channels, out_channels, kernel_size=5, stride=stride, padding=2, output_padding=1)

def conv_resblock_three(in_channels, out_channels, stride=1):
    return nn.Sequential(conv3x3(in_channels, out_channels, stride), nn.ReLU(), ResBlock(out_channels), ResBlock(out_channels), ResBlock(out_channels))

def conv_resblock_two(in_channels, out_channels, stride=1): 
    return nn.Sequential(conv3x3(in_channels, out_channels, stride), nn.ReLU(), ResBlock(out_channels), ResBlock(out_channels))

def conv_resblock_one(in_channels, out_channels, stride=1):
    return nn.Sequential(conv3x3(in_channels, out_channels, stride), nn.ReLU(), ResBlock(out_channels))

def conv_1x1_resblock_one(in_channels, out_channels, stride=1):
    return nn.Sequential(conv1x1(in_channels, out_channels, stride), nn.ReLU(), ResBlock(out_channels))

def conv_resblock_two_DS(in_channels, out_channels, stride=2):
    return nn.Sequential(conv3x3(in_channels, out_channels, stride), nn.ReLU(), ResBlock(out_channels), ResBlock(out_channels))

def conv3x3_leaky_relu(in_channels, out_channels, stride=1):
    return nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=True), nn.LeakyReLU(0.1))

def conv1x1_leaky_relu(in_channels, out_channels, stride=1):
    return nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0, bias=True), nn.LeakyReLU(0.1))