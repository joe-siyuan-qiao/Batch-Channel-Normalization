import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.nn import functional as F
from torch.autograd import Function


class Conv2d(nn.Conv2d):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        super(Conv2d, self).__init__(in_channels, out_channels, kernel_size, stride,
                 padding, dilation, groups, bias)

    def forward(self, x):
        weight = self.weight
        weight_mean = weight.mean(dim=1, keepdim=True).mean(dim=2,
                                  keepdim=True).mean(dim=3, keepdim=True)
        weight = weight - weight_mean
        std = torch.pow(weight.view(weight.size(0), -1).var(dim=1) + 1e-5, 0.5).view(-1, 1, 1, 1)
        weight = weight / std.expand_as(weight)
        return F.conv2d(x, weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)


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

