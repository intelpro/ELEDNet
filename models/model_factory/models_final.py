import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
from collections import namedtuple
from string import Template
from torch.autograd import Function
from torch.nn.modules.utils import _pair
from torchvision.ops import DeformConv2d
from thop import profile
from einops import rearrange
import cupy
import numbers
from models.submodules import *


def conv1x1(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0, bias=True)


def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=True)


def conv5x5(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=5, stride=stride, padding=2, bias=True)


def deconv4x4(in_channels, out_channels, stride=2):
    return nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=stride, padding=1)


def deconv5x5(in_channels, out_channels, stride=2):
    return nn.ConvTranspose2d(in_channels, out_channels, kernel_size=5, stride=stride, padding=2, output_padding=1)

def conv(in_channels, out_channels, kernel_size, bias=False, stride = 1):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias, stride = stride)


class ResBlock(nn.Module):
    def __init__(self, inplanes, planes, kernel_size=3, stride=1, dilation=1, groups=1):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=kernel_size, stride=stride,
                               padding=get_same_padding(kernel_size, dilation), dilation=dilation, groups=groups)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=kernel_size, stride=1,
                               padding=get_same_padding(kernel_size, dilation), dilation=dilation, groups=groups)
        self.relu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        self.res_translate = None
        if not inplanes == planes or not stride == 1:
            self.res_translate = nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride)


    def forward(self, x):
        residual = x

        out = self.relu(self.conv1(x))
        out = self.conv2(out)

        if self.res_translate is not None:
            residual = self.res_translate(residual)
        out += residual

        return out

class DownSample(nn.Module):
    def __init__(self, in_channels, s_factor):
        super(DownSample, self).__init__()
        self.down = nn.Sequential(nn.Upsample(scale_factor=0.5, mode='bilinear', align_corners=False),
                                  nn.Conv2d(in_channels, in_channels + s_factor, 1, stride=1, padding=0, bias=False))

    def forward(self, x):
        x = self.down(x)
        return x


class UpSample(nn.Module):
    def __init__(self, in_channels, s_factor):
        super(UpSample, self).__init__()
        self.up = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                                nn.Conv2d(in_channels + s_factor, in_channels, 1, stride=1, padding=0, bias=False))

    def forward(self, x):
        x = self.up(x)
        return x


class SkipUpSample(nn.Module):
    def __init__(self, in_channels, s_factor):
        super(SkipUpSample, self).__init__()
        self.up = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                                nn.Conv2d(in_channels + s_factor, in_channels, 1, stride=1, padding=0, bias=False))

    def forward(self, x, y):
        x = self.up(x)
        x = x + y
        return x


# Channel Attention Layer
class CALayer(nn.Module):
    def __init__(self, channel, reduction=16, bias=False):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=bias),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=bias),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y

## Channel Attention Block (CAB)
class CAB(nn.Module):
    def __init__(self, n_feat, kernel_size, reduction, bias, act):
        super(CAB, self).__init__()
        modules_body = []
        modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))
        modules_body.append(act)
        modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))

        self.CA = CALayer(n_feat, reduction, bias=bias)
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res = self.CA(res)
        res += x
        return res

## Original Resolution Block (ORB)
class CABs(nn.Module):
    def __init__(self, n_feat, kernel_size, reduction, act, bias, num_cab):
        super(CABs, self).__init__()
        modules_body = []
        modules_body = [CAB(n_feat, kernel_size, reduction, bias=bias, act=act) for _ in range(num_cab)]
        modules_body.append(conv(n_feat, n_feat, kernel_size))
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res += x
        return res

# RDB-based RNN cell
class shallow_cell(nn.Module):
    def __init__(self, n_feat):
        super(shallow_cell, self).__init__()
        self.n_feats = n_feat
        act = nn.PReLU()
        bias = False
        reduction = 4
        self.shallow_feat = nn.Sequential(conv(3, self.n_feats, 3, bias=bias),
                                           CAB(self.n_feats, 3, reduction, bias=bias, act=act))

    def forward(self,x):
        feat = self.shallow_feat(x)
        return feat


# RDB-based RNN cell
class shallow_cell_events(nn.Module):
    def __init__(self, n_feat):
        super(shallow_cell_events, self).__init__()
        self.n_feats = n_feat
        act = nn.PReLU()
        bias = False
        reduction = 4
        self.shallow_feat = nn.Sequential(conv(16, self.n_feats, 3, bias=bias),
                                           CAB(self.n_feats, 3, reduction, bias=bias, act=act))

    def forward(self,x):
        feat = self.shallow_feat(x)
        return feat

def conv_down(in_chn, out_chn, bias=False):
    layer = nn.Conv2d(in_chn, out_chn, kernel_size=4, stride=2, padding=1, bias=bias)
    return layer


def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight

class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)
        assert len(normalized_shape) == 1
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias

class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)

Stream = namedtuple('Stream', ['ptr'])

def Dtype(t):
    if isinstance(t, torch.cuda.FloatTensor):
        return 'float'
    elif isinstance(t, torch.cuda.DoubleTensor):
        return 'double'


# @cupy._util.memoize(for_each_device=True)
def load_kernel(kernel_name, code, **kwargs):
    code = Template(code).substitute(**kwargs)
    kernel_code = cupy.cuda.compile_with_cache(code)
    return kernel_code.get_function(kernel_name)


CUDA_NUM_THREADS = 1024

kernel_loop = '''
#define CUDA_KERNEL_LOOP(i, n)                        \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; \
      i < (n);                                       \
      i += blockDim.x * gridDim.x)
'''


def GET_BLOCKS(N):
    return (N + CUDA_NUM_THREADS - 1) // CUDA_NUM_THREADS


_idynamic_kernel = kernel_loop + '''
extern "C"
__global__ void idynamic_forward_kernel(
const ${Dtype}* bottom_data, const ${Dtype}* weight_data, ${Dtype}* top_data) {
  CUDA_KERNEL_LOOP(index, ${nthreads}) {
    const int n = index / ${channels} / ${top_height} / ${top_width};
    const int c = (index / ${top_height} / ${top_width}) % ${channels};
    const int h = (index / ${top_width}) % ${top_height};
    const int w = index % ${top_width};
    const int g = c / (${channels} / ${groups});
    ${Dtype} value = 0;
    #pragma unroll
    for (int kh = 0; kh < ${kernel_h}; ++kh) {
      #pragma unroll
      for (int kw = 0; kw < ${kernel_w}; ++kw) {
        const int h_in = -${pad_h} + h * ${stride_h} + kh * ${dilation_h};
        const int w_in = -${pad_w} + w * ${stride_w} + kw * ${dilation_w};
        if ((h_in >= 0) && (h_in < ${bottom_height})
          && (w_in >= 0) && (w_in < ${bottom_width})) {
          const int offset = ((n * ${channels} + c) * ${bottom_height} + h_in)
            * ${bottom_width} + w_in;
          const int offset_weight = ((((n * ${groups} + g) * ${kernel_h} + kh) * ${kernel_w} + kw) * ${top_height} + h)
            * ${top_width} + w;
          value += weight_data[offset_weight] * bottom_data[offset];
        }
      }
    }
    top_data[index] = value;
  }
}
'''

_idynamic_kernel_backward_grad_input = kernel_loop + '''
extern "C"
__global__ void idynamic_backward_grad_input_kernel(
    const ${Dtype}* const top_diff, const ${Dtype}* const weight_data, ${Dtype}* const bottom_diff) {
  CUDA_KERNEL_LOOP(index, ${nthreads}) {
    const int n = index / ${channels} / ${bottom_height} / ${bottom_width};
    const int c = (index / ${bottom_height} / ${bottom_width}) % ${channels};
    const int h = (index / ${bottom_width}) % ${bottom_height};
    const int w = index % ${bottom_width};
    const int g = c / (${channels} / ${groups});
    ${Dtype} value = 0;
    #pragma unroll
    for (int kh = 0; kh < ${kernel_h}; ++kh) {
      #pragma unroll
      for (int kw = 0; kw < ${kernel_w}; ++kw) {
        const int h_out_s = h + ${pad_h} - kh * ${dilation_h};
        const int w_out_s = w + ${pad_w} - kw * ${dilation_w};
        if (((h_out_s % ${stride_h}) == 0) && ((w_out_s % ${stride_w}) == 0)) {
          const int h_out = h_out_s / ${stride_h};
          const int w_out = w_out_s / ${stride_w};
          if ((h_out >= 0) && (h_out < ${top_height})
                && (w_out >= 0) && (w_out < ${top_width})) {
            const int offset = ((n * ${channels} + c) * ${top_height} + h_out)
                  * ${top_width} + w_out;
            const int offset_weight = ((((n * ${groups} + g) * ${kernel_h} + kh) * ${kernel_w} + kw) * ${top_height} + h_out)
                  * ${top_width} + w_out;
            value += weight_data[offset_weight] * top_diff[offset];
          }
        }
      }
    }
    bottom_diff[index] = value;
  }
}
'''

_idynamic_kernel_backward_grad_weight = kernel_loop + '''
extern "C"
__global__ void idynamic_backward_grad_weight_kernel(
    const ${Dtype}* const top_diff, const ${Dtype}* const bottom_data, ${Dtype}* const buffer_data) {
  CUDA_KERNEL_LOOP(index, ${nthreads}) {
    const int h = (index / ${top_width}) % ${top_height};
    const int w = index % ${top_width};
    const int kh = (index / ${kernel_w} / ${top_height} / ${top_width})
          % ${kernel_h};
    const int kw = (index / ${top_height} / ${top_width}) % ${kernel_w};
    const int h_in = -${pad_h} + h * ${stride_h} + kh * ${dilation_h};
    const int w_in = -${pad_w} + w * ${stride_w} + kw * ${dilation_w};
    if ((h_in >= 0) && (h_in < ${bottom_height})
          && (w_in >= 0) && (w_in < ${bottom_width})) {
      const int g = (index / ${kernel_h} / ${kernel_w} / ${top_height} / ${top_width}) % ${groups};
      const int n = (index / ${groups} / ${kernel_h} / ${kernel_w} / ${top_height} / ${top_width}) % ${num};
      ${Dtype} value = 0;
      #pragma unroll
      for (int c = g * (${channels} / ${groups}); c < (g + 1) * (${channels} / ${groups}); ++c) {
        const int top_offset = ((n * ${channels} + c) * ${top_height} + h)
              * ${top_width} + w;
        const int bottom_offset = ((n * ${channels} + c) * ${bottom_height} + h_in)
              * ${bottom_width} + w_in;
        value += top_diff[top_offset] * bottom_data[bottom_offset];
      }
      buffer_data[index] = value;
    } else {
      buffer_data[index] = 0;
    }
  }
}
'''

class _idynamic(Function):
    @staticmethod
    def forward(ctx, input, weight, stride, padding, dilation):
        assert input.dim() == 4 and input.is_cuda
        assert weight.dim() == 6 and weight.is_cuda
        batch_size, channels, height, width = input.size()
        kernel_h, kernel_w = weight.size()[2:4]
        output_h = int((height + 2 * padding[0] - (dilation[0] * (kernel_h - 1) + 1)) / stride[0] + 1)
        output_w = int((width + 2 * padding[1] - (dilation[1] * (kernel_w - 1) + 1)) / stride[1] + 1)

        output = input.new(batch_size, channels, output_h, output_w)
        n = output.numel()

        with torch.cuda.device_of(input):
            f = load_kernel('idynamic_forward_kernel', _idynamic_kernel, Dtype=Dtype(input), nthreads=n,
                            num=batch_size, channels=channels, groups=weight.size()[1],
                            bottom_height=height, bottom_width=width,
                            top_height=output_h, top_width=output_w,
                            kernel_h=kernel_h, kernel_w=kernel_w,
                            stride_h=stride[0], stride_w=stride[1],
                            dilation_h=dilation[0], dilation_w=dilation[1],
                            pad_h=padding[0], pad_w=padding[1])
            f(block=(CUDA_NUM_THREADS, 1, 1),
              grid=(GET_BLOCKS(n), 1, 1),
              args=[input.data_ptr(), weight.data_ptr(), output.data_ptr()],
              stream=Stream(ptr=torch.cuda.current_stream().cuda_stream))

        ctx.save_for_backward(input, weight)
        ctx.stride, ctx.padding, ctx.dilation = stride, padding, dilation
        return output

    @staticmethod
    def backward(ctx, grad_output):
        assert grad_output.is_cuda
        if not grad_output.is_contiguous():
            grad_output.contiguous()
        input, weight = ctx.saved_tensors
        stride, padding, dilation = ctx.stride, ctx.padding, ctx.dilation

        batch_size, channels, height, width = input.size()
        kernel_h, kernel_w = weight.size()[2:4]
        output_h, output_w = grad_output.size()[2:]

        grad_input, grad_weight = None, None

        opt = dict(Dtype=Dtype(grad_output),
                   num=batch_size, channels=channels, groups=weight.size()[1],
                   bottom_height=height, bottom_width=width,
                   top_height=output_h, top_width=output_w,
                   kernel_h=kernel_h, kernel_w=kernel_w,
                   stride_h=stride[0], stride_w=stride[1],
                   dilation_h=dilation[0], dilation_w=dilation[1],
                   pad_h=padding[0], pad_w=padding[1])

        with torch.cuda.device_of(input):
            if ctx.needs_input_grad[0]:
                grad_input = input.new(input.size())

                n = grad_input.numel()
                opt['nthreads'] = n

                f = load_kernel('idynamic_backward_grad_input_kernel',
                                _idynamic_kernel_backward_grad_input, **opt)
                f(block=(CUDA_NUM_THREADS, 1, 1),
                  grid=(GET_BLOCKS(n), 1, 1),
                  args=[grad_output.data_ptr(), weight.data_ptr(), grad_input.data_ptr()],
                  stream=Stream(ptr=torch.cuda.current_stream().cuda_stream))

            if ctx.needs_input_grad[1]:
                grad_weight = weight.new(weight.size())

                n = grad_weight.numel()
                opt['nthreads'] = n

                f = load_kernel('idynamic_backward_grad_weight_kernel',
                                _idynamic_kernel_backward_grad_weight, **opt)
                f(block=(CUDA_NUM_THREADS, 1, 1),
                  grid=(GET_BLOCKS(n), 1, 1),
                  args=[grad_output.data_ptr(), input.data_ptr(), grad_weight.data_ptr()],
                  stream=Stream(ptr=torch.cuda.current_stream().cuda_stream))

        return grad_input, grad_weight, None, None, None


def _idynamic_cuda(input, weight, bias=None, stride=1, padding=0, dilation=1):
    """ idynamic kernel
    """
    assert input.size(0) == weight.size(0)
    assert input.size(-2) // stride == weight.size(-2)
    assert input.size(-1) // stride == weight.size(-1)
    if input.is_cuda:
        out = _idynamic.apply(input, weight, _pair(stride), _pair(padding), _pair(dilation))
        if bias is not None:
            out += bias.view(1, -1, 1, 1)
    else:
        raise NotImplementedError
    return out




###############################
# ResNet
###############################


def get_same_padding(kernel_size, dilation):
    kernel_size = kernel_size + (kernel_size - 1) * (dilation - 1)
    padding = (kernel_size - 1) // 2
    return padding


class ResBlock(nn.Module):
    def __init__(self, inplanes, planes, kernel_size=3, stride=1, dilation=1, groups=1):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=kernel_size, stride=stride,
                               padding=get_same_padding(kernel_size, dilation), dilation=dilation, groups=groups)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=kernel_size, stride=1,
                               padding=get_same_padding(kernel_size, dilation), dilation=dilation, groups=groups)
        self.relu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        self.res_translate = None
        if not inplanes == planes or not stride == 1:
            self.res_translate = nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride)


    def forward(self, x):
        residual = x

        out = self.relu(self.conv1(x))
        out = self.conv2(out)

        if self.res_translate is not None:
            residual = self.res_translate(residual)
        out += residual

        return out


class Encoder(nn.Module):
    def __init__(self, n_feat, scale_unetfeats, kernel_size=3, reduction=4, bias=False):
        super(Encoder, self).__init__()
        act = nn.PReLU()
        self.encoder_level1 = [CAB(n_feat, kernel_size, reduction, bias=bias, act=act) for _ in range(2)]
        self.encoder_level2 = [CAB(n_feat + scale_unetfeats, kernel_size, reduction, bias=bias, act=act) for _ in range(2)]
        self.encoder_level3 = [CAB(n_feat + (scale_unetfeats * 2), kernel_size, reduction, bias=bias, act=act) for _ in range(2)]

        self.encoder_level1 = nn.Sequential(*self.encoder_level1)
        self.encoder_level2 = nn.Sequential(*self.encoder_level2)
        self.encoder_level3 = nn.Sequential(*self.encoder_level3)

        self.down12 = DownSample(n_feat, scale_unetfeats)
        self.down23 = DownSample(n_feat + scale_unetfeats, scale_unetfeats)

    def forward(self, x):
        ### level 1
        enc1 = self.encoder_level1(x)
        x = self.down12(enc1)
        ### level 2
        enc2 = self.encoder_level2(x)
        x = self.down23(enc2)
        ### level 3
        enc3 = self.encoder_level3(x)
        return [enc1, enc2, enc3]


class IDynamicDWConv(nn.Module):
    def __init__(self, channels, kernel_size, group_channels, down, conv_group):
        super(IDynamicDWConv, self).__init__()
        self.kernel_size = kernel_size
        self.channels = channels
        self.group_channels = group_channels
        self.down = down
        self.groups = self.channels // self.group_channels
        self.avgpool = nn.AvgPool2d(kernel_size=down, stride=down)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        Block1 = [ResBlock(channels, channels, kernel_size=kernel_size, stride=1, groups=conv_group) for _ in range(3)]
        Block2 = [ResBlock(channels, channels, kernel_size=kernel_size, stride=1, groups=conv_group) for _ in range(3)]
        self.tokernel = nn.Conv2d(channels, kernel_size**2*self.groups, 1, 1, 0)
        self.Block1 = nn.Sequential(*Block1)
        self.Block2 = nn.Sequential(*Block2)

    def forward(self, x, y):
        weight = self.tokernel(self.Block2(self.maxpool(self.Block1(self.avgpool(y)))))
        weight = F.interpolate(weight, scale_factor=2*self.down, mode='bilinear')
        b, c, h, w = weight.shape
        weight = weight.view(b, self.groups, self.kernel_size, self.kernel_size, h, w)
        out = _idynamic_cuda(x, weight, stride=1, padding=(self.kernel_size - 1) // 2)
        return out
    



class ffn_align(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(ffn_align, self).__init__()
        hidden_features = int(dim*ffn_expansion_factor)
        self.project_in_f = nn.Conv3d(dim, hidden_features, kernel_size=(3, 1, 1), stride=1, padding=(1, 0, 0), bias=bias)
        self.project_in_ev = nn.Conv3d(dim, hidden_features, kernel_size=(3, 1, 1), stride=1, padding=(1, 0, 0), bias=bias)
        ##
        self.conv1_prev = nn.Conv2d(hidden_features*3, hidden_features, 3, 1, 1, bias=bias)
        self.conv1_future = nn.Conv2d(hidden_features*3, hidden_features, 3, 1, 1, bias=bias)
        self.resblock_forward = ResidualBlocks2D(hidden_features, 3)
        self.resblock_backward = ResidualBlocks2D(hidden_features, 3)
        self.conv_prop = nn.Conv2d(hidden_features*3, hidden_features, 3, 1, 1,bias=bias)
        self.resblock_prop = ResidualBlocks2D(hidden_features, 3)
        ## align-ment
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x_f, x_ev):
        bs, t, c, h, w = x_f.shape
        x_f = self.project_in_f(x_f)
        x_ev = self.project_in_ev(x_ev)
        x_f = rearrange(x_f, 'b c t h w -> b t c h w', b=bs)
        x_ev = rearrange(x_ev, 'b c t h w -> b t c h w', b=bs)
        ### propagation
        ## prev
        x_f_prev = x_f[:, 0]
        x_ev_prev = torch.cat((x_ev[:, 0], x_ev[:, 1]), dim=1)
        input_prev = torch.cat((x_f_prev, x_ev_prev), dim=1)
        prop_prev = self.conv1_prev(input_prev)
        prop_prev = self.resblock_forward(prop_prev)
        ## future
        x_f_future = x_f[:, 2]
        x_ev_future = torch.cat((x_ev[:, 1], x_ev[:, 2]), dim=1)
        input_future = torch.cat((x_f_future, x_ev_future), dim=1)
        prop_future = self.conv1_future(input_future)
        prop_future = self.resblock_backward(prop_future)
        ## cur
        x_f_cur = x_f[:, 1]
        prop_input = torch.cat((x_f_cur, prop_future, prop_prev), dim=1)
        prop_out = self.conv_prop(prop_input)
        prop_out = self.resblock_prop(prop_out)
        return prop_out
    
class alignment_layer(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias, LayerNorm_type):
        super(alignment_layer, self).__init__()
        self.norm_frame = LayerNorm(dim, LayerNorm_type)
        self.norm_event = LayerNorm(dim, LayerNorm_type)
        self.alignment_ffn = ffn_align(dim, ffn_expansion_factor, bias)
    
    def forward(self, x_f, x_ev):
        b = x_f.shape[0]
        x_f = self.norm_frame(rearrange(x_f, 'b t c h w -> (b t) c h w', b=b))
        x_ev = self.norm_event(rearrange(x_ev, 'b t c h w -> (b t) c h w', b=b))
        x_f_re = rearrange(x_f, '(b t) c h w -> b c t h w', b=b)
        x_ev_re = rearrange(x_ev, '(b t) c h w -> b c t h w', b=b)
        aligned_feat = self.alignment_ffn(x_f_re, x_ev_re)
        return aligned_feat


##########################################################################
## Gated-Dconv Feed-Forward Network (GDFN)
class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()
        hidden_features = int(dim*ffn_expansion_factor)
        self.project_in = nn.Conv2d(dim, hidden_features*2, kernel_size=1, bias=bias)
        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3, stride=1, 
                                padding=1, groups=hidden_features*2, bias=bias)
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x



##########################################################################
## Multi-DConv Head Transposed Self-Attention (MDTA)
class Attention(nn.Module):
    def __init__(self, dim, num_heads, stride, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.stride = stride
        self.qk = nn.Conv2d(dim, dim*2, kernel_size=1, bias=bias)
        self.qk_dwconv = nn.Conv2d(dim*2, dim*2, kernel_size=3, stride=self.stride, padding=1, groups=dim*2, bias=bias)

        self.v = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.v_dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)

        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        
    def forward(self, x):
        b,c,h,w = x.shape

        qk = self.qk_dwconv(self.qk(x))
        q,k = qk.chunk(2, dim=1)
        
        v = self.v_dwconv(self.v(x))
        
        b, f, h1, w1 = q.size()

        q = rearrange(q, 'b (head c) h1 w1 -> b head c (h1 w1)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h1 w1 -> b head c (h1 w1)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)
        
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out
        

class alignment(nn.Module):
    def __init__(self, dim, dim_prev=None, memory=False, stride=1, type='group_conv'):
        super(alignment, self).__init__()
        act = nn.GELU()
        bias = False
        kernel_size = 3
        padding = kernel_size//2
        deform_groups = 8
        out_channels = deform_groups * 3 * (kernel_size**2)
        ## fw
        self.fw_offset_conv = nn.Conv2d(dim, out_channels, kernel_size, stride=1, padding=padding, bias=bias)
        self.fw_conv1 = nn.Conv2d(dim*4, dim, 3, 1, 1, bias=bias)
        self.fw_bottleneck = nn.Sequential(nn.Conv2d(dim, dim, kernel_size = 3, padding = 1, bias = bias), act)
        self.fw_deform = DeformConv2d(dim, dim, kernel_size, padding = 2, groups = deform_groups, dilation=2)            
        ## bw
        self.bw_offset_conv = nn.Conv2d(dim, out_channels, kernel_size, stride=1, padding=padding, bias=bias)
        self.bw_conv1 = nn.Conv2d(dim*4, dim, 3, 1, 1, bias=bias)
        self.bw_bottleneck = nn.Sequential(nn.Conv2d(dim, dim, kernel_size = 3, padding = 1, bias = bias), act)
        self.bw_deform = DeformConv2d(dim, dim, kernel_size, padding = 2, groups = deform_groups, dilation=2)
        ## alignment final
        self.align_final = nn.Conv2d(dim*2, dim, 3, 1, 1, bias=bias)
        ## memory offset
        if memory==True:
            ## fw offset
            self.fw_offset_feat_up = nn.ConvTranspose2d(dim_prev, dim, 3, stride=2, padding=1, output_padding=1)
            self.bw_offset_feat_up = nn.ConvTranspose2d(dim_prev, dim, 3, stride=2, padding=1, output_padding=1)
            self.fw_bottleneck_prev = nn.Sequential(nn.Conv2d(dim*2, dim, kernel_size = 3 , padding=1, bias=bias), act)
            self.bw_bottleneck_prev = nn.Sequential(nn.Conv2d(dim*2, dim, kernel_size = 3 , padding=1, bias=bias), act)
            
    def offset_gen(self, x):
        o1, o2, mask = torch.chunk(x, 3, dim=1)
        offset = torch.cat((o1, o2), dim=1)
        mask = torch.sigmoid(mask)
        return offset, mask
        
    def forward(self, x, x_ev, bs, prev_fw_offset_feat=None, prev_bw_offset_feat=None):
        ##
        x = rearrange(x, '(b t) c h w -> b t c h w', b=bs)
        x_ev = rearrange(x_ev, '(b t) c h w -> b t c h w', b=bs)
        ##
        x_prev = x[:, 0]
        x_cur = x[:, 1]
        x_future = x[:, 2]
        ### forward prop
        x_ev_prev = torch.cat((x_ev[:, 0], x_ev[:, 1]), dim=1)
        input_prev = torch.cat((x_prev, x_cur, x_ev_prev), dim=1)
        prop_prev = self.fw_conv1(input_prev)
        fw_offset_feat = self.fw_bottleneck(prop_prev)
        ## forward memory
        if prev_fw_offset_feat is not None:
            prev_fw_offset_feat = self.fw_offset_feat_up(prev_fw_offset_feat)
            fw_offset_feat = self.fw_bottleneck_prev(torch.cat((fw_offset_feat, prev_fw_offset_feat), dim=1))
        fw_offset, fw_mask = self.offset_gen(self.fw_offset_conv(fw_offset_feat)) 
        prop_prev = self.fw_deform(prop_prev, fw_offset, fw_mask)
        ### backward prop
        x_ev_future = torch.cat((x_ev[:, 1], x_ev[:, 2]), dim=1)
        input_future = torch.cat((x_cur, x_future, x_ev_future), dim=1)
        prop_future = self.bw_conv1(input_future)
        bw_offset_feat = self.bw_bottleneck(prop_future)
        ## bw memory
        if prev_bw_offset_feat is not None:
            prev_bw_offset_feat = self.fw_offset_feat_up(prev_bw_offset_feat)
            bw_offset_feat = self.fw_bottleneck_prev(torch.cat((bw_offset_feat, prev_bw_offset_feat), dim=1))
        bw_offset, bw_mask = self.offset_gen(self.bw_offset_conv(bw_offset_feat)) 
        prop_future = self.bw_deform(prop_future, bw_offset, bw_mask)
        ##
        aligned_feat = self.align_final(torch.cat((prop_prev, prop_future), dim=1))
        return aligned_feat, fw_offset_feat, bw_offset_feat


class Transformer(nn.Module):
    def __init__(self, dim, num_heads, stride, ffn_expansion_factor, bias, LayerNorm_type):
        super(Transformer, self).__init__()
        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, stride, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x



class EDTFA(nn.Module):
    def __init__(self, in_channels=48):
        super(EDTFA, self).__init__()
        heads = [1,2,4,8]
        bias = False
        LayerNorm_type = 'WithBias'
        base_channel = 32
        scale_unet_feat = base_channel
        ## down and encoder
        self.down1 = nn.Conv2d(base_channel, base_channel + scale_unet_feat, 3, stride=2, padding=1)        
        self.down2 = nn.Conv2d(base_channel+ scale_unet_feat, base_channel+2*scale_unet_feat, 3, stride=2, padding=1)
        self.encoder_level1 = nn.Sequential(*[Transformer(dim=base_channel, num_heads=heads[0], stride=1,
                                                  ffn_expansion_factor=2.66, bias=bias, 
                                                  LayerNorm_type=LayerNorm_type) for i in range(2)])
        self.encoder_level2 = nn.Sequential(*[Transformer(dim=base_channel + scale_unet_feat, num_heads=heads[1], stride=1, 
                                                  ffn_expansion_factor=2.66, bias=bias, 
                                                  LayerNorm_type=LayerNorm_type) for i in range(2)])
        ## event - encoder
        self.down1_e = nn.Conv2d(base_channel, base_channel + scale_unet_feat, 3, stride=2, padding=1)
        self.down2_e = nn.Conv2d(base_channel + scale_unet_feat, base_channel + 2*scale_unet_feat , 3, stride=2, padding=1)
        self.encoder_level1_e = nn.Sequential(*[Transformer(dim=base_channel, num_heads=heads[0], stride=1 ,ffn_expansion_factor=2.66, 
                                                    bias=bias, LayerNorm_type=LayerNorm_type) for i in range(2)])
        self.encoder_level2_e = nn.Sequential(*[Transformer(dim=base_channel + scale_unet_feat, num_heads=heads[1], stride=1, 
                                                    ffn_expansion_factor=2.66, bias=bias, 
                                                    LayerNorm_type=LayerNorm_type) for i in range(2)])
        self.alignment0 = alignment(base_channel, base_channel+scale_unet_feat, memory=True)
        self.alignment1 = alignment(base_channel + scale_unet_feat, base_channel + 2*scale_unet_feat, memory=True)
        self.alignment2 = alignment(base_channel + 2*scale_unet_feat)

        self.up1 = nn.ConvTranspose2d(base_channel + scale_unet_feat, 
                                      base_channel, 3, stride=2, padding=1, output_padding=1)        
        self.up2 = nn.ConvTranspose2d(base_channel + 2*scale_unet_feat, 
                                      base_channel + scale_unet_feat, 3, stride=2, padding=1, output_padding=1)
        
    def forward(self, x, x_event, bs):
        ## frame
        x = self.encoder_level1(x)
        enc1_f = self.down1(x)
        enc1_f = self.encoder_level2(enc1_f)
        enc2_f = self.down2(enc1_f)
        ### event
        x_event = self.encoder_level1_e(x_event)
        enc1_e = self.down1_e(x_event)
        enc1_e = self.encoder_level2_e(enc1_e)
        enc2_e = self.down2_e(enc1_e)
        ## alignment
        enc2, fw_offset_fw2, bw_offset_bw2 = self.alignment2(enc2_f, enc2_e, bs)
        dec1 = self.up2(enc2)
        enc1_f, fw_offset_fw1, bw_offset_bw1  = self.alignment1(enc1_f, enc1_e, bs, fw_offset_fw2, bw_offset_bw2)
        dec1 = dec1 + enc1_f
        dec0 = self.up1(dec1)
        x, _, _ = self.alignment0(x, x_event, bs, fw_offset_fw1, bw_offset_bw1)
        x = x + dec0
        return x


class Decoder(nn.Module):
    def __init__(self, n_feat, scale_unetfeats, kernel_size=3, reduction=4, bias=False):
        super(Decoder, self).__init__()
        act = nn.PReLU()
        self.decoder_level1 = [CAB(n_feat, kernel_size, reduction, bias=bias, act=act) for _ in range(2)]
        self.decoder_level2 = [CAB(n_feat + scale_unetfeats, kernel_size, reduction, bias=bias, act=act) for _ in range(2)]
        self.decoder_level3 = [CAB(n_feat + (scale_unetfeats * 2), kernel_size, reduction, bias=bias, act=act) for _ in range(2)]

        self.decoder_level1 = nn.Sequential(*self.decoder_level1)
        self.decoder_level2 = nn.Sequential(*self.decoder_level2)
        self.decoder_level3 = nn.Sequential(*self.decoder_level3)

        self.skip_attn1 = CAB(n_feat, kernel_size, reduction, bias=bias, act=act)
        self.skip_attn2 = CAB(n_feat + scale_unetfeats, kernel_size, reduction, bias=bias, act=act)

        self.up21 = SkipUpSample(n_feat, scale_unetfeats)
        self.up32 = SkipUpSample(n_feat + scale_unetfeats, scale_unetfeats)
        ## reconstruction
        self.recons_1 = conv5x5(n_feat, 3)
        self.recons_2 = conv5x5(n_feat+scale_unetfeats, 3)
        self.recons_3 = conv5x5(n_feat+2*scale_unetfeats, 3)

    def forward(self, outs):
        enc1, enc2, enc3 = outs
        dec3 = self.decoder_level3(enc3)
        deblurred_img_0 = self.recons_3(dec3)
        deblurred_img_0 = torch.clamp(deblurred_img_0, 0, 1)

        x = self.up32(dec3, self.skip_attn2(enc2))
        dec2 = self.decoder_level2(x)
        deblurred_img_1 = self.recons_2(dec2)
        deblurred_img_1 = torch.clamp(deblurred_img_1, 0, 1)

        x = self.up21(dec2, self.skip_attn1(enc1))
        dec1 = self.decoder_level1(x)
        deblurred_img_2 = self.recons_1(dec1)
        deblurred_img_2 = torch.clamp(deblurred_img_2, 0, 1)
        return [deblurred_img_2, deblurred_img_1, deblurred_img_0]

#### fusion module ###
class Frequency_module(nn.Module):
    def __init__(self, num_channels):
        super(Frequency_module, self).__init__()
        self.conv1_y = nn.Conv2d(2 * num_channels, 2 * num_channels, kernel_size=1, stride=1, padding=0)
        self.conv2_y = nn.Conv2d(2 * num_channels, 2 * num_channels, kernel_size=1, stride=1, padding=0)

        self.conv1_z = nn.Conv2d(2 * num_channels, 2 * num_channels, kernel_size=1, stride=1, padding=1)
        self.conv2_z = nn.Conv2d(2 * num_channels, 2 * num_channels, kernel_size=1, stride=1, padding=1)
        ## sigma
        self.sigma = 7

    def make_gaussian(self, y_idx, x_idx, height, width):
        yv, xv = torch.meshgrid([torch.arange(0, height), torch.arange(0, width)])
        yv = yv.unsqueeze(0).float().cuda()
        xv = xv.unsqueeze(0).float().cuda()
        g = torch.exp(- ((yv - y_idx) ** 2 + (xv - x_idx) ** 2) / (2 * self.sigma** 2))
        return g.unsqueeze(0)       #1, 1, H, W

    def forward(self, x):
        b, c, h, w = x.shape
        x = x.float()
        y = torch.fft.fft2(x)

        h_idx, w_idx = h // 2, w // 2
        high_filter = self.make_gaussian(h_idx, w_idx, h, w)
        ## high frequency regions
        f = y * high_filter

        y_imag = f.imag
        y_real = f.real
        y_f = torch.cat([y_real, y_imag], dim=1)
        f = F.relu(self.conv1_y(y_f))

        f = self.conv2_y(f).float()
        f_real, f_imag = torch.chunk(f, 2, dim=1)
        f = torch.complex(f_real, f_imag)

        f = torch.fft.ifft2(f, s=(h, w)).float()
        high_out = x+f
        return high_out


class FeedForward_bottom_level(nn.Module):
    def __init__(self, dim, dim_prev, ffn_expansion_factor, bias):
        super(FeedForward_bottom_level, self).__init__()
        hidden_features = int(dim*ffn_expansion_factor)
        self.num_feat = hidden_features
        self.deconv_hidden = deconv4x4(dim_prev, hidden_features)
        self.project_f = conv3x3(dim, hidden_features)
        self.project_e = conv3x3(dim, hidden_features)
        self.project_cat_feat = conv3x3_leaky_relu(hidden_features*3, hidden_features*2)
        self.conv1_1 = conv3x3_leaky_relu(2*hidden_features, hidden_features)
        self.conv1_2 = conv3x3_leaky_relu(2*hidden_features, hidden_features)
        ## dynamic filter
        self.freq = Frequency_module(hidden_features)
        self.dp = IDynamicDWConv(hidden_features,  3, 1, 2, 2)
        self.cat_conv = conv3x3_leaky_relu(hidden_features, hidden_features*2)
        self.conv2 = nn.Conv2d(hidden_features, hidden_features, 3, 1, 1, bias=True)
        self.conv3 = nn.Conv2d(hidden_features, hidden_features, 3, 1, 1, bias=True)
        ## output projection
        self.project_CAB = CABs(hidden_features, 3, reduction=4, act = nn.PReLU(), bias=False, num_cab=3)
        self.project_out = conv3x3_leaky_relu(hidden_features, dim)

    def forward(self, x_f, x_ev, fusion_feat_prev):
        deconv_fusion_feat = self.deconv_hidden(fusion_feat_prev)
        x_f = self.project_f(x_f)
        x_ev = self.project_e(x_ev)
        ## cat feat
        cat_feat = torch.cat((x_f, x_ev, deconv_fusion_feat), dim=1)
        cat_feat = self.project_cat_feat(cat_feat)
        ## split
        cat_feat1 = self.conv1_1(cat_feat)
        cat_feat2 = self.conv1_2(cat_feat)
        # cat_feat1, cat_feat2 = torch.split(cat_feat, self.num_feat, dim=1)
        ## high frequency
        high_feat = self.freq(cat_feat1)
        high_feat = self.dp(high_feat, high_feat)
        high_feat = high_feat * torch.sigmoid(self.conv2(high_feat))
        ## org feat
        cat_feat2 = cat_feat2 * torch.sigmoid(self.conv3(cat_feat2))
        cat_feat = high_feat + cat_feat2
        ## dp
        cat_feat = self.project_CAB(cat_feat)
        x = self.project_out(cat_feat)
        return x

class ImageEventFusion_bottom_level(nn.Module):
    def __init__(self, dim, dim_prev, ffn_expansion_factor, bias, LayerNorm_type):
        super(ImageEventFusion_bottom_level, self).__init__()
        self.ffn = FeedForward_bottom_level(dim, dim_prev, ffn_expansion_factor, bias)

    def forward(self, x_f, x_ev, fusion_feat_prev):
        x = self.ffn(x_f, x_ev, fusion_feat_prev)
        return x

class FeedForward_top_level(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward_top_level, self).__init__()
        hidden_features = int(dim*ffn_expansion_factor)
        self.num_feat = hidden_features
        ### modules
        self.project_f = conv3x3(dim, hidden_features)
        self.project_e = conv3x3(dim, hidden_features)
        self.project_cat_feat = conv3x3_leaky_relu(hidden_features*2, hidden_features*2)
        self.conv1_1 = conv3x3_leaky_relu(2*hidden_features, hidden_features)
        self.conv1_2 = conv3x3_leaky_relu(2*hidden_features, hidden_features)
        self.freq = Frequency_module(hidden_features)
        self.dp = IDynamicDWConv(hidden_features,  3, 1, 2, 2)
        self.cat_conv = conv3x3_leaky_relu(hidden_features, hidden_features*2)
        self.conv2 = nn.Conv2d(hidden_features, hidden_features, 3, 1, 1, bias=True)
        self.conv3 = nn.Conv2d(hidden_features, hidden_features, 3, 1, 1, bias=True)
        self.project_CAB = CABs(hidden_features, 3, reduction=4, act = nn.PReLU(), bias=False, num_cab=3)
        self.project_out = conv3x3_leaky_relu(hidden_features, dim)

    def forward(self, x_f, x_ev):
        x_f = self.project_f(x_f)
        x_ev = self.project_e(x_ev)
        ## cat feat
        cat_feat = torch.cat((x_f, x_ev), dim=1)
        cat_feat = self.project_cat_feat(cat_feat)
        ## split
        cat_feat1 = self.conv1_1(cat_feat)
        cat_feat2 = self.conv1_2(cat_feat)
        ## high frequency
        high_feat = self.freq(cat_feat1)
        high_feat = self.dp(high_feat, high_feat)
        high_feat = high_feat * torch.sigmoid(self.conv2(high_feat))
        ## org feat
        cat_feat2 = cat_feat2 * torch.sigmoid(self.conv3(cat_feat2))
        cat_feat = high_feat + cat_feat2
        ## output feature processing
        cat_feat = self.project_CAB(cat_feat)
        x = self.project_out(cat_feat)
        return x

class ImageEventFusion_top_level(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias, LayerNorm_type):
        super(ImageEventFusion_top_level, self).__init__()
        self.ffn = FeedForward_top_level(dim, ffn_expansion_factor, bias)

    def forward(self, x_f, x_ev):
        x = self.ffn(x_f, x_ev)
        return x


class EventDeblurNet(nn.Module):
    def __init__(self):
        super(EventDeblurNet, self).__init__()
        ### ev down convolution
        base_feat = 32
        scale_unet_feat = base_feat
        self.device = torch.device('cuda')
        # RNN cell
        self.shallow_cell_frames = shallow_cell(n_feat=base_feat)
        self.shallow_cell_events = shallow_cell_events(n_feat=base_feat)
        self.conv_fusion = conv1x1(2*base_feat, base_feat)
        self.encoder_frame = Encoder(n_feat=base_feat, scale_unetfeats=scale_unet_feat)
        self.encoder_event = Encoder(n_feat=base_feat, scale_unetfeats=scale_unet_feat)
        # decoder
        self.decoder = Decoder(n_feat=base_feat, scale_unetfeats=scale_unet_feat)
        ## recon id
        self.recon_id = 1
        self.scale_range = 3
        ## alignment
        self.align = EDTFA(base_feat)
        ## alignment module
        ## fusion module
        fusion_list = []
        fusion_list.append(ImageEventFusion_bottom_level(base_feat, int(base_feat + scale_unet_feat), 
                                                         ffn_expansion_factor=1, bias=False, LayerNorm_type='WithBias'))
        fusion_list.append(ImageEventFusion_bottom_level(int(base_feat + scale_unet_feat), int(base_feat + 2*scale_unet_feat),
                                                          ffn_expansion_factor=1.25, bias=False, LayerNorm_type='WithBias'))
        fusion_list.append(ImageEventFusion_top_level(int(base_feat + 2*scale_unet_feat),
                                                       ffn_expansion_factor=1.5, bias=False, LayerNorm_type='WithBias'))
        self.fusion_list = nn.ModuleList(fusion_list)

    def forward(self, batch):
        b, t, c, h, w = batch['blur_input_clip'].shape
        x_frame = batch['blur_input_clip']
        x_event = batch['event_vox_clip']
        ## frame feature processing
        x_frame = rearrange(x_frame, 'b t c h w -> (b t) c h w')
        f_feature = self.shallow_cell_frames(x_frame)
        ## event feature processing
        x_event = rearrange(x_event, 'b t c h w -> (b t) c h w')
        ## event feature
        e_feature = self.shallow_cell_events(x_event)
        ### alignment
        aligned_feat = self.align(f_feature, e_feature, b)
        # ## frame encoding
        f_encoder_outs = self.encoder_frame(aligned_feat)
        ## event encoding
        e_encoder_outs = self.encoder_event(e_feature)
        # ## align module
        # # prop_hidden = torch.new_zeros(b, c, h, w)
        for scale_idx in range(self.scale_range-1, -1, -1):
            e_encoder_outs[scale_idx] = rearrange(e_encoder_outs[scale_idx], '(b t) c h w -> b t c h w', b=b)
        # ## fusion
        fusion_out_0 = self.fusion_list[-1](f_encoder_outs[-1], e_encoder_outs[-1][:, 1])
        fusion_out_1 = self.fusion_list[-2](f_encoder_outs[-2], e_encoder_outs[-2][:, 1], fusion_out_0)
        fusion_out_2 = self.fusion_list[-3](f_encoder_outs[-3], e_encoder_outs[-3][:, 1], fusion_out_1)
        # ##
        encoder_outs = []
        encoder_outs.append(fusion_out_2)
        encoder_outs.append(fusion_out_1)
        encoder_outs.append(fusion_out_0)
        # ##################
        # #       TFR      #
        # ##################
        output_dict = self.decoder(encoder_outs)
        return output_dict