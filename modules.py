import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.nn import functional as F
import spikingjelly.clock_driven.neuron as neuron
import spikingjelly.clock_driven.ann2snn.modules as spike_module

class SpikingNorm(nn.Module):
    def __init__(self, momentum=0.1, scale=True, sigmoid=True ,eps=1e-6):
        super(SpikingNorm, self).__init__()
        self.sigmoid = sigmoid
        self.eps = eps
        self.lock_max = False
        if scale:
            self.scale = Parameter(torch.Tensor([1.0]))
        else:
            self.register_buffer('scale', torch.ones(1))
        if self.sigmoid:
            self.scale.data *= 10.0
        self.momentum = momentum
        self.register_buffer('running_max', torch.ones(1))

    def calc_scale(self):
        if self.sigmoid:
            scale = torch.sigmoid(self.scale)
        else:
            scale = torch.abs(self.scale)
        return scale

    def calc_v_th(self):
        return self.running_max * self.calc_scale() + self.eps

    def forward(self, x):
        if self.training and (not self.lock_max):
            self.running_max = (1 - self.momentum) * self.running_max + self.momentum * torch.max(F.relu(x)).item()
        x = torch.clamp( F.relu(x) / self.calc_v_th(), min=0.0, max=1.0)
        # x = F.relu(x)
        return x

    def extra_repr(self):
        if self.sigmoid:
            return 'v_th={}, scale={}, running_max={}'.format(
                self.calc_v_th(), torch.sigmoid(self.scale.data), self.running_max.data
            )
        else:
            return 'v_th={}, scale={}, running_max={}'.format(
                self.calc_v_th(), torch.abs(self.scale.data), self.running_max.data
            )

    def extract_running_max(self):
        x = self.running_max.data
        self.running_max.data = self.running_max.data / x
        return x

    def extract_scale(self):
        x = self.scale.data
        self.running_max.data = self.running_max.data * x
        self.scale.data = torch.Tensor([1.0]).to(self.scale.device)
        return x

class SpikingNormInv(nn.Module):
    def __init__(self, momentum=0.1, scale=True, sigmoid=True ,eps=1e-6):
        super(SpikingNormInv, self).__init__()
        self.sigmoid = sigmoid
        self.eps = eps
        self.lock_max = False
        if scale:
            self.scale = Parameter(torch.Tensor([1.0]))
        else:
            self.register_buffer('scale', torch.ones(1))
        if self.sigmoid:
            self.scale.data *= 10.0
        self.momentum = momentum
        self.register_buffer('running_max', torch.ones(1))

    def calc_scale(self):
        if self.sigmoid:
            scale = torch.sigmoid(self.scale)
        else:
            scale = torch.abs(self.scale)
        return scale

    def calc_v_th(self):
        return (self.running_max + self.eps) / self.calc_scale()

    def forward(self, x):
        if self.training and (not self.lock_max):
            self.running_max = (1 - self.momentum) * self.running_max + self.momentum * torch.max(F.relu(x)).item()
        x = torch.clamp( F.relu(x) / self.calc_v_th(), min=0.0, max=1.0)
        # x = F.relu(x)
        return x

    def extra_repr(self):
        if self.sigmoid:
            return 'v_th={}, scale={}, running_max={}'.format(
                self.calc_v_th(), torch.sigmoid(self.scale.data), self.running_max.data
            )
        else:
            return 'v_th={}, scale={}, running_max={}'.format(
                self.calc_v_th(), torch.abs(self.scale.data), self.running_max.data
            )

    def extract_running_max(self):
        x = self.running_max.data
        self.running_max.data = self.running_max.data / x
        return x

    def extract_scale(self):
        x = self.scale.data
        self.running_max.data = self.running_max.data / x
        self.scale.data = torch.Tensor([1.0]).to(self.scale.device)
        return x


def replace_relu_by_spikingnorm_inv(model,scale=True,share_scale=None):
    for name, module in model._modules.items():
        if hasattr(module,"_modules"):
            model._modules[name] = replace_relu_by_spikingnorm_inv(module,scale,share_scale)
        if 'relu' in module.__class__.__name__.lower():
            if not scale:
                model._modules[name] = SpikingNormInv(scale=False)
            else:
                if share_scale is None:
                    model._modules[name] = SpikingNormInv()
                else:
                    model._modules[name] = SpikingNormInv()
                    model._modules[name].scale = share_scale
    return model

def replace_relu_by_spikingnorm(model,scale=True,share_scale=None):
    for name, module in model._modules.items():
        if hasattr(module,"_modules"):
            model._modules[name] = replace_relu_by_spikingnorm(module,scale,share_scale)
        if 'relu' in module.__class__.__name__.lower():
            if not scale:
                model._modules[name] = SpikingNorm(scale=False)
            else:
                if share_scale is None:
                    model._modules[name] = SpikingNorm()
                else:
                    model._modules[name] = SpikingNorm()
                    model._modules[name].scale = share_scale
    return model

def replace_maxpool2d_by_avgpool2d(model):
    for name, module in model._modules.items():
        if hasattr(module,"_modules"):
            model._modules[name] = replace_maxpool2d_by_avgpool2d(module)
        if module.__class__.__name__ == 'MaxPool2d':
            model._modules[name] = nn.AvgPool2d(kernel_size=module.kernel_size,
                                                stride=module.stride,
                                                padding=module.padding)
    return model

def replace_maxpool2d_by_spikemaxpool(model):
    for name, module in model._modules.items():
        if hasattr(module,"_modules"):
            model._modules[name] = replace_maxpool2d_by_spikemaxpool(module)
        if module.__class__.__name__ == 'MaxPool2d':
            model._modules[name] = spike_module.MaxPool2d(kernel_size=module.kernel_size,
                                                stride=module.stride,
                                                padding=module.padding)
    return model

class ClippedReLU(nn.Module):
    def __init__(self):
        super(ClippedReLU, self).__init__()

    def forward(self, x):
        return torch.clamp(F.relu(x), min=0.0, max=1.0)

def replace_relu_by_clippedrelu(model):
    for name, module in model._modules.items():
        if hasattr(module,"_modules"):
            model._modules[name] = replace_relu_by_clippedrelu(module)
        if module.__class__.__name__ == 'ReLU':
            model._modules[name] = ClippedReLU()
    return model

def remove_batchnorm(model):
    for name, module in model._modules.items():
        if hasattr(module,"_modules"):
            model._modules[name] = remove_batchnorm(module)
        if "BatchNorm" in module.__class__.__name__:
            model._modules[name] = nn.Sequential()
    return model

def replace_spikingnorm_by_ifnode(model):
    for name, module in model._modules.items():
        if hasattr(module,"_modules"):
            model._modules[name] = replace_spikingnorm_by_ifnode(module)
        if module.__class__.__name__ == "SpikingNorm":
            model._modules[name] = neuron.IFNode(v_threshold=module.calc_v_th().data,v_reset=None)
    return model

def replace_relu_by_ifnode(model):
    for name, module in model._modules.items():
        if hasattr(module,"_modules"):
            model._modules[name] = replace_relu_by_ifnode(module)
        if module.__class__.__name__ == "ReLU":
            model._modules[name] = neuron.IFNode(v_reset=None)
    return model

def replace_ifnode_by_relu(model):
    for name, module in model._modules.items():
        if hasattr(module,"_modules"):
            model._modules[name] = replace_ifnode_by_relu(module)
        if module.__class__.__name__ == "IFNode":
            model._modules[name] = nn.ReLU()
    return model

# def replace_spikingnorm_by_scaledifnode(model):
#     for name, module in model._modules.items():
#         if hasattr(module,"_modules"):
#             model._modules[name] = replace_spikingnorm_by_scaledifnode(module)
#         if module.__class__.__name__ == "SpikingNorm":
#             model._modules[name] = neuron.ScaledIFNode(v_threshold=module.calc_v_th().data*0.75,emit=module.calc_v_th().data,v_reset=None)
#     return model

def replace_ifnode_by_spikingnorm(model,sigmoid=False):
    for name, module in model._modules.items():
        if hasattr(module,"_modules"):
            model._modules[name] = replace_ifnode_by_spikingnorm(module,sigmoid)
        if module.__class__.__name__ == "IFNode":
            model._modules[name] = SpikingNorm(sigmoid=sigmoid)
    return model

def remove_runningmax_from_spikingnorm(model):
    last_param_module = None
    for name, module in model.named_modules():
        if isinstance(module,(nn.Linear,nn.Conv2d)):
            last_param_module = module
        if isinstance(module,SpikingNorm):
            if last_param_module is not None:
                if hasattr(last_param_module, 'weight'):
                    running_max = module.extract_running_max()
                    last_param_module.weight.data = last_param_module.weight.data / running_max
                    if hasattr(last_param_module,'bias') and last_param_module.bias is not None:
                        last_param_module.bias.data = last_param_module.bias.data / running_max
            last_param_module = None
    return model

def replace_spikingnorm_by_relu(model, scale=1.0):
    for name, module in model._modules.items():
        if hasattr(module,"_modules"):
            model._modules[name] = replace_spikingnorm_by_relu(module, scale)
        if module.__class__.__name__ == "SpikingNorm":
            model._modules[name] = nn.ReLU()
    return model


if __name__ == "__main__":
    pass
    # print(torch.sigmoid(torch.FloatTensor([5])))
    # exit(-1)
    # # pass
    # # x = torch.tensor([[0.2,0.3],[0.4,0.5]])
    # # scale = 0.2
    # #
    # # def layerwise_loss(a):
    # #     #print(torch.pow(torch.norm(a, 2), 2))
    # #     k = torch.sum(a) / (torch.pow(torch.norm(a, 2), 2))
    # #     return k
    # # print(layerwise_loss(torch.clamp(x/scale,0,1)))
    # # print(torch.clamp(x/scale,0,1))
    #
    # import spikingjelly.clock_driven.neuron as neuron
    # import spikingjelly.clock_driven.functional as functional
    # import numpy as np
    # import matplotlib.pyplot as plt
    #
    # n = neuron.ScaledIFNode(v_threshold=5.0,v_reset=None)
    #
    # y = []
    # a = []
    # h = np.linspace(-1, 10, 100)
    # for i in h:
    #     x = torch.FloatTensor([i])
    #     functional.reset_net(n)
    #     sum = 0
    #     for t in range(100):
    #         s = n(x)
    #         # print(s)
    #         sum += s
    #     sum /= 100
    #     # print(sum)
    #     a.append(x.item())
    #     y.append(sum.item())
    # plt.plot(a,y)
    # plt.show()
    #
    #
    # h2 = torch.from_numpy(h)
    # x = 5.0 * torch.clamp(h2 / 5.0,0,1)
    # plt.plot(a,x)
    # plt.show()