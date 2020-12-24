#conding=utf-8
# @Time  : 2020/6/3 15:14
# @Author: fufuyu
# @Email:  fufuyu@tencent.com

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.parameter import Parameter
from .loss import normalize

class GeneralizedMeanPooling(nn.Module):
    r"""Applies a 2D power-average adaptive pooling over an input signal composed of several input planes.
    The function computed is: :math:`f(X) = pow(sum(pow(X, p)), 1/p)`
        - At p = infinity, one gets Max Pooling
        - At p = 1, one gets Average Pooling
    The output is of size H x W, for any input size.
    The number of output features is equal to the number of input planes.
    Args:
        output_size: the target output size of the image of the form H x W.
                     Can be a tuple (H, W) or a single H for a square image H x H
                     H and W can be either a ``int``, or ``None`` which means the size will
                     be the same as that of the input.
    """

    def __init__(self, norm, output_size=1, eps=1e-6):
        super(GeneralizedMeanPooling, self).__init__()
        assert norm > 0
        self.p = float(norm)
        self.output_size = output_size
        self.eps = eps

    def forward(self, x):
        x = x.clamp(min=self.eps).pow(self.p)
        return torch.nn.functional.adaptive_avg_pool2d(x, self.output_size).pow(1. / self.p)

    def __repr__(self):
        return self.__class__.__name__ + '(' \
               + str(self.p) + ', ' \
               + 'output_size=' + str(self.output_size) + ')'


class GeneralizedMeanPoolingP(GeneralizedMeanPooling):
    """ Same, but norm is trainable
    """

    def __init__(self, norm=3, output_size=1, eps=1e-6):
        super(GeneralizedMeanPoolingP, self).__init__(norm, output_size, eps)
        self.p = nn.Parameter(torch.ones(1) * norm)


class NonLocal(nn.Module):
    def __init__(self, in_channels, inter_channels=None, bn_layer=False):
        super(NonLocal, self).__init__()


        self.in_channels = in_channels
        self.inter_channels = inter_channels

        if self.inter_channels is None:
            self.inter_channels = in_channels // 4
            if self.inter_channels == 0:
                self.inter_channels = 1


        self.g = nn.Conv2d(in_channels=self.in_channels,
                         out_channels=self.inter_channels,
                         kernel_size=1, stride=1, padding=0) #, bias=False

        if bn_layer:
            self.W = nn.Sequential(
                nn.Conv2d(in_channels=self.inter_channels,
                          out_channels=self.in_channels,
                        kernel_size=1, stride=1, padding=0), #, bias=False
                nn.BatchNorm2d(self.in_channels)
            )
            nn.init.constant_(self.W[1].weight, 0)
            nn.init.constant_(self.W[1].bias, 0)
        else:
            self.W = nn.Conv2d(in_channels=self.inter_channels, out_channels=self.in_channels,
                             kernel_size=1, stride=1, padding=0) #, bias=False
            nn.init.constant_(self.W.weight, 0)
            # nn.init.constant_(self.W.bias, 0)

        self.theta = nn.Conv2d(in_channels=self.in_channels,
                               out_channels=self.inter_channels,
                              kernel_size=1, stride=1, padding=0) #, bias=False
        self.phi = nn.Conv2d(in_channels=self.in_channels,
                              out_channels=self.inter_channels,
                              kernel_size=1, stride=1, padding=0) #, bias=False

    def forward(self, x):
        '''
        :param x: (b, c, t, h, w)
        :return:
        '''

        batch_size = x.size(0)

        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)
        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)
        f = torch.matmul(theta_x, phi_x)
        f_div_C = F.softmax(f, dim=-1)

        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        W_y = self.W(y)
        z = W_y + x

        return z


class PairGraph(nn.Module):
    def __init__(self, in_channels, inter_channels=None):
        super(PairGraph, self).__init__()


        self.in_channels = in_channels
        self.inter_channels = inter_channels

        if self.inter_channels is None:
            self.inter_channels = in_channels // 16
            if self.inter_channels == 0:
                self.inter_channels = 1


        self.y = nn.Conv2d(in_channels=self.in_channels,
                         out_channels=self.inter_channels,
                         kernel_size=1, stride=1, padding=0, bias=False)

        self.g = nn.Conv2d(in_channels=self.in_channels,
                               out_channels=self.inter_channels,
                              kernel_size=1, stride=1, padding=0, bias=False)
        self.o = nn.Conv2d(in_channels=self.inter_channels,
                              out_channels=self.in_channels,
                              kernel_size=1, stride=1, padding=0, bias=False)

        nn.init.constant_(self.o.weight, 0)
        # nn.init.constant_(self.o.bias, 0)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        '''
        :param x: (b, c, t, h, w)
        :return:
        '''

        b, c, h, w = x.size()

        y = self.y(x)
        y = normalize(y, axis=1)   # n, c, h, w
        y = y.reshape(b, self.inter_channels, -1)  # n, c, h*w
        y1 = y.permute(0, 2, 1)   # n, h*w, c

        coef = self.relu(y1.matmul(y))    # n, h*w, h*w

        batch_eye = torch.eye(h*w, device=x.device).reshape(1, h*w, h*w).repeat(b, 1, 1)
        adja = coef * (1. - batch_eye)
        degree = torch.sum(adja, dim=-1, keepdim=True) * batch_eye
        # flag = degree > 1e-3
        temp = torch.where(degree > 1e-3, degree, torch.ones_like(degree, device=x.device))
        # temp = flag * degree + (1 - flag) * torch.ones_like(degree, device=x.device)
        norm = temp.rsqrt() * batch_eye
        norm = norm.detach()
        f = torch.matmul(torch.matmul(norm, degree - adja), norm)   # n, h*w, h*w

        g = self.g(x).reshape(b, self.inter_channels, -1)  # n, c, h*w
        o = torch.matmul(g, f).reshape(b, self.inter_channels, h, w)
        o = self.o(o)
        return x + o


class DSBN2d(nn.Module):
    def __init__(self, planes):
        super(DSBN2d, self).__init__()
        self.num_features = planes
        self.BN_S = nn.BatchNorm2d(planes)
        self.BN_T = nn.BatchNorm2d(planes)

    def forward(self, x):
        if (not self.training):
            return self.BN_S(x)

        bs = x.size(0)
        assert (bs%2==0)
        split = torch.split(x, int(bs/2), 0)
        out1 = self.BN_S(split[0].contiguous())
        out2 = self.BN_T(split[1].contiguous())
        out = torch.cat((out1, out2), 0)
        return out

class DSBN2dConstBatch(nn.Module):
    def __init__(self, planes, batch_size=64, constant_batch=32):
        super(DSBN2dConstBatch, self).__init__()
        self.num_features = planes
        self.constant_batch = constant_batch
        self.bn_list = nn.Sequential(*[nn.BatchNorm2d(planes) for _ in range(batch_size // constant_batch)])
    def forward(self, x):
        if (not self.training):
            return self.bn_list[0](x)

        bs = x.size(0)
        # print('befor', bs, x.size(), self.constant_batch)
        assert (bs % self.constant_batch==0)
        split = torch.split(x, self.constant_batch, 0)
        out_list = [self.bn_list[i](split[i].contiguous()) for i in range(bs // self.constant_batch)]
        out = torch.cat(out_list, 0)
        # print('after', bs, out.size(), self.constant_batch, len(out_list), out_list[0].size())
        return out

class DSBN2dShare(nn.Module):
    def __init__(self, planes, constant_batch=32):
        super(DSBN2dShare, self).__init__()
        self.num_features = planes
        self.constant_batch = constant_batch
        self.bn = nn.BatchNorm2d(planes)
    def forward(self, x):
        if (not self.training):
            return self.bn(x)

        bs = x.size(0)
        # print(bs, self.constant_batch)
        assert (bs % self.constant_batch==0)
        split = torch.split(x, self.constant_batch, 0)
        out_list = [self.bn(split[i].contiguous()) for i in range(bs // self.constant_batch)]
        out = torch.cat(out_list, 0)
        return out

def convert_dsbn(model):
    for _, (child_name, child) in enumerate(model.named_children()):
        # print(next(model.parameters()))
        # assert(not next(model.parameters()).is_cuda)
        if isinstance(child, nn.BatchNorm2d):
            m = DSBN2d(child.num_features)
            m.BN_S.load_state_dict(child.state_dict())
            m.BN_T.load_state_dict(child.state_dict())
            setattr(model, child_name, m)
        else:
            convert_dsbn(child)

def convert_dsbnShare(model, constant_batch=32):
    for _, (child_name, child) in enumerate(model.named_children()):
        # print(next(model.parameters()))
        # assert(not next(model.parameters()).is_cuda)
        if isinstance(child, nn.BatchNorm2d):
            m = DSBN2dShare(child.num_features, constant_batch=constant_batch)
            m.bn.load_state_dict(child.state_dict())
            setattr(model, child_name, m)
        else:
            convert_dsbnShare(child, constant_batch=constant_batch)

def convert_dsbnConstBatch(model, batch_size=64, constant_batch=32):
    for _, (child_name, child) in enumerate(model.named_children()):
        # print(next(model.parameters()))
        # assert(not next(model.parameters()).is_cuda)
        if isinstance(child, nn.BatchNorm2d):
            m = DSBN2dConstBatch(child.num_features, batch_size=batch_size, constant_batch=constant_batch)
            for bn in m.bn_list:
                bn.load_state_dict(child.state_dict())
            setattr(model, child_name, m)
        else:
            convert_dsbnConstBatch(child, batch_size=batch_size, constant_batch=constant_batch)



def BatchNorm2d(num_features):
    num_groups = 32
    return GN(num_channels=num_features, num_groups=num_groups)


class GN(nn.Module):

    def __init__(self, num_channels, num_groups):
        super(GN, self).__init__()
        self.num_channels = num_channels
        self.num_groups = num_groups
        self.weight = Parameter(torch.ones(1, num_groups, 1))
        self.bias = Parameter(torch.zeros(1, num_groups, 1))
        self.pbn = nn.BatchNorm2d(num_channels)

    def forward(self, inp):
        out = self.pbn(inp)
        out = out.view(1, inp.size(0) * self.num_groups, -1)
        out = torch.batch_norm(out, None, None, None, None, True, 0, 1e-5, True)
        out = out.view(inp.size(0), self.num_groups, -1)
        out = self.weight * out + self.bias
        out = out.view_as(inp)
        return out