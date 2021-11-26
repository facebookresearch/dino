'''Adapted from the transformer_net.py example on the Pytorch github repo.

Link:
https://github.com/pytorch/examples/blob/0c1654d6913f77f09c0505fb284d977d89c17c1a/fast_neural_style/neural_style/transformer_net.py
'''
import torch
import torch.nn as nn
from torch.nn import functional as F, init
from utils import *
import dotmap
import random
import torch_dct as dct

ACTIVATIONS = {
    'relu': torch.nn.ReLU,
    'leaky_relu': torch.nn.LeakyReLU,
    'gelu': torch.nn.LeakyReLU # TODO: change.
}

class Img2Img(torch.nn.Module):
    '''Network that maps an image and noise vector to another image of the same size.'''
    def __init__(self, filter_size, noise_dim, num_channels=3, residual=False, bounded=False, L1_forced=False, L2=False, L2_forced=False, half=False, L12_forced=False, L12_elastic_forced=False,
                 bound_magnitude=0.05, budget_type = "all", neutralad=False, vary_bound=False, affine_layer=False, color_layer=False, warping_layer=False, 
                 unconditional=False, wb_layer=False, kernel_layer=False, noise_dist=None, activation='relu', network='basic', clamp=True, spectral=False, downsample_to=False, num_res_blocks=5, double=False):
        super().__init__()
        if residual and bounded:
            raise RuntimeError('Only one of residual and bounded may be True.')
        
        self.num_channels = num_channels
        self.filter_size = filter_size
        self.num_res_blocks = num_res_blocks  # Number of residual layers if network==basic.

        self.noise_dist = noise_dist
        self.activation = activation
        self.network = network
        self.vary_bound = vary_bound
        self.clamp = clamp
        self.spectral = spectral
        self.downsample_to = downsample_to # Rescale to downsample_to x downsample_to before input into viewmaker, then upsample
        self.double = double # Whether to generate two views at once
        if self.double:
            self.num_channels *= 2
        
        # Initial convolution layers (+ 1 for noise filter)
        self.conv1 = ConvLayer(self.num_channels + 1, 32, kernel_size=9, stride=1)
        self.in1 = torch.nn.InstanceNorm2d(32, affine=True)
        self.conv2 = ConvLayer(32, 64, kernel_size=3, stride=2)
        self.in2 = torch.nn.InstanceNorm2d(64, affine=True)
        self.conv3 = ConvLayer(64, 128, kernel_size=3, stride=2)
        self.in3 = torch.nn.InstanceNorm2d(128, affine=True)
        
        # Residual layers +N for added random channels
        self.res1 = ResidualBlock(128 + 1)
        self.res2 = ResidualBlock(128 + 2)
        self.res3 = ResidualBlock(128 + 3)
        self.res4 = ResidualBlock(128 + 4)
        self.res5 = ResidualBlock(128 + 5)
        
        # Upsampling Layers
        self.deconv1 = UpsampleConvLayer(
            128 + self.num_res_blocks, 64, kernel_size=3, stride=1, upsample=2)
        self.in4 = torch.nn.InstanceNorm2d(64, affine=True)
        self.deconv2 = UpsampleConvLayer(
            64, 32, kernel_size=3, stride=1, upsample=2)
        self.in5 = torch.nn.InstanceNorm2d(32, affine=True)
        if warping_layer:
            # 2 components for displacement field
            self.deconv3 = ConvLayer(32, self.num_channels + 2, kernel_size=9, stride=1)
        else:
            # 3 channels for warping 
            self.deconv3 = ConvLayer(32, self.num_channels, kernel_size=9, stride=1)
        
        # Non-linearities
        self.act = ACTIVATIONS[self.activation]()
        self.residual = residual
        self.bounded = bounded
        self.L1_forced = L1_forced
        self.L2_forced = L2_forced
        self.L2 = L2
        self.L12_forced = L12_forced
        self.L12_elastic_forced = L12_elastic_forced
        self.half = half
        self.bound_magnitude = bound_magnitude
        self.budget_type = budget_type
        self.neutralad = neutralad
        self.affine_layer = affine_layer
        self.color_layer = color_layer
        self.warping_layer = warping_layer
        self.wb_layer = wb_layer
        self.kernel_layer = kernel_layer

        print(f"Set up viewmaker model with bound magnitude: {self.bound_magnitude}, budget type: {self.budget_type}, neutralad: {self.neutralad}")

        # initialize the weights to be 0
        if residual or bounded: self.apply(self.zero_init)

        if self.affine_layer:
            self.affine = nn.Linear(128, 6)  # Affine matrix (6)
        
        if self.color_layer:
            self.color = nn.Linear(128, self.num_channels)  # Channel jitter (3)
        
        if self.wb_layer:
            self.wb = nn.Linear(128, 2)  # Weight and bias
        
        if self.kernel_layer:
            self.kernel = nn.Linear(128, 9)  # 3x3 Kernel

        self.unconditional = unconditional

        if self.network == 'batch_norm':
            self.add_batch_norm_params()
        if self.network == 'small':
            self.add_smaller_params()

    @staticmethod
    def zero_init(m):
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            # actual 0 has symmetry problems
            init.normal_(m.weight.data, mean=0, std=1e-4)
            # init.constant_(m.weight.data, 0)
            init.constant_(m.bias.data, 0)
        elif isinstance(m, nn.BatchNorm1d):
            pass

    def add_noise_channel(self, x, num=1, bound_multiplier=1):
        # bound_multiplier is a scalar or a 1D tensor of length batch_size
        batch_size = x.size(0)
        filter_size = x.size(-1)
        shp = (batch_size, num, filter_size, filter_size)
        if not self.noise_dist or self.noise_dist == 'uniform':
            bound_multiplier = torch.tensor(bound_multiplier, device=x.device)
            noise = torch.rand(shp, device=x.device) * bound_multiplier.view(-1, 1, 1, 1)
        elif self.noise_dist == 'gaussian':
            noise = torch.normal(torch.zeros(shp), device=x.device) * bound_multiplier  # Gaussian noise.
        else:
            raise ValueError(f'Undefined noise distribution {self.noise_dist}')
        return torch.cat((x, noise), dim=1)

    def basic_net(self, y, num_res_blocks=5, bound_multiplier=1):
        if num_res_blocks not in list(range(6)):
            raise ValueError(f'num_res_blocks must be in {list(range(6))}, got {num_res_blocks}.')

        y = self.add_noise_channel(y, bound_multiplier=bound_multiplier)
        y = self.act(self.in1(self.conv1(y)))
        y = self.act(self.in2(self.conv2(y)))
        y = self.act(self.in3(self.conv3(y)))

        # [batch_size, 128]
        features = y.clone().mean([-1, -2])
        
        for i, res in enumerate([self.res1, self.res2, self.res3, self.res4, self.res5]):
            if i < num_res_blocks:
                y = res(self.add_noise_channel(y, bound_multiplier=bound_multiplier))

        y = self.act(self.in4(self.deconv1(y)))
        y = self.act(self.in5(self.deconv2(y)))
        y = self.deconv3(y)

        return y, features
    
    def smaller_net(self, y):
        y = self.add_noise_channel(y)
        y = self.act(self.sm_in1(self.sm_conv1(y)))

        # [batch_size, 32]
        features = y.clone().mean([-1, -2])

        y = self.sm_res1(self.add_noise_channel(y))

        # y = self.act(self.sm_in2(self.sm_deconv1(y)))
        y = self.sm_deconv2(y)

        return y, features

    def add_smaller_params(self):
        self.sm_conv1 = ConvLayer(self.num_channels + 1, 8, kernel_size=3, stride=1)
        self.sm_in1 = torch.nn.InstanceNorm2d(8)
        self.sm_res1 = ResidualBlock(8+1)
        # self.sm_deconv1 = ConvLayer(32 + 1, 32, kernel_size=3, stride=1)
        # self.sm_in2 = torch.nn.InstanceNorm2d(32)
        self.sm_deconv2 = ConvLayer(8 + 1, self.num_channels, kernel_size=3, stride=1)

    def add_batch_norm_params(self):
        self.bn1 = torch.nn.BatchNorm2d(32)
        self.bn2 = torch.nn.BatchNorm2d(64)
        self.bn3 = torch.nn.BatchNorm2d(128)
        self.bn4 = torch.nn.BatchNorm2d(64)
        self.bn5 = torch.nn.BatchNorm2d(32)

        self.bn_res1 = torch.nn.BatchNorm2d(128+1)
        self.bn_res2 = torch.nn.BatchNorm2d(128+2)
        self.bn_res3 = torch.nn.BatchNorm2d(128+3)
        self.bn_res4 = torch.nn.BatchNorm2d(128+4)
        self.bn_res5 = torch.nn.BatchNorm2d(128+5)

    def batchnorm_net(self, y):
        y = self.add_noise_channel(y)
        y = self.act(self.bn1(self.conv1(y)))
        y = self.act(self.bn2(self.conv2(y)))
        y = self.act(self.bn3(self.conv3(y)))

        # [batch_size, 128]
        features = y.clone().mean([-1, -2])
        
        y = self.bn_res1(self.res1(self.add_noise_channel(y)))
        y = self.bn_res2(self.res2(self.add_noise_channel(y)))
        y = self.bn_res3(self.res3(self.add_noise_channel(y)))
        y = self.bn_res4(self.res4(self.add_noise_channel(y)))
        y = self.bn_res5(self.res5(self.add_noise_channel(y)))

        y = self.act(self.bn4(self.deconv1(y)))
        y = self.act(self.bn5(self.deconv2(y)))
        y = self.deconv3(y)

        return y, features
    
    def apply_kernel(self, x, kernel):

        kernel = kernel / kernel.norm(dim=-1, keepdim=True) # Enforce common norm.
        kernel = kernel.view(-1, 1, 3, 3).repeat(1, self.num_channels, 1, 1)  # Repeat for channels.
        # x has size [batch_size, C, W, H]
        # kernel has size [batch_size, C, W, H]
        return F.conv2d(x, kernel, groups=x.size(0))

    def get_delta(self, y_pixels, bound_multiplier=1):
        '''Produces constrained perturbation.'''
        bound_magnitude = self.bound_magnitude
        if self.neutralad:
            delta = torch.tanh(y_pixels)  # Project to [-1, 1]
            if self.budget_type == "all": 
                # Scale all deltas down
                avg_magnitude = delta.abs().mean([1, 2, 3], keepdim=True)
                max_magnitude = bound_magnitude
                delta = delta * max_magnitude / (avg_magnitude + 1e-4)
            elif self.budget_type == "partial":
                # Scale down the deltas with high norms - use partial budget
                avg_magnitude = delta.abs().mean([1, 2, 3], keepdim=True)
                max_magnitude = bound_magnitude
                delta = torch.where(delta > max_magnitude, delta * max_magnitude / (avg_magnitude + 1e-4), delta)
            elif self.budget_type == "none":
                pass
            else:
                raise Exception(f"Budget type {self.budget_type} is not supported")
            return delta

        if self.vary_bound:
            # Becomes 1D Tensor if bound_multiplier is 1D Tensor 
            # (reshaped for [batch_size, channels, height, width])
            bound_magnitude = bound_magnitude * bound_multiplier.reshape(-1, 1, 1, 1)
        if self.residual:
            # Average of original image and generated one.
            # result = (torch.sigmoid(y) + x) / 2.
            delta = bound_magnitude * torch.tanh(y_pixels)
        elif self.bounded:
            delta = torch.tanh(y_pixels)
            avg_magnitude = delta.abs().mean([1,2,3], keepdim=True)
            # scale down average change if too big.
            max_magnitude = bound_magnitude
            delta = torch.where(avg_magnitude > max_magnitude, delta * max_magnitude / (avg_magnitude + 2e-4), delta)
            # inv_sigmoid_x = torch.log(x / (1-x))
            # result = torch.sigmoid(x + delta)
        elif self.L1_forced:
            delta = torch.tanh(y_pixels)
            avg_magnitude = delta.abs().mean([1,2,3], keepdim=True)
            # scale down average change if too big.
            max_magnitude = bound_magnitude
            delta = delta * max_magnitude / (avg_magnitude + 1e-4)
        elif self.L2:
            delta = torch.tanh(y_pixels)
            avg_magnitude = torch.sqrt( (delta ** 2).mean([1, 2, 3], keepdim=True) )
            # scale down average change if too big.
            max_magnitude = bound_magnitude
            delta = torch.where(avg_magnitude > max_magnitude,
                                delta * max_magnitude / avg_magnitude, delta)
        elif self.L2_forced:
            delta = torch.tanh(y_pixels)
            avg_magnitude = torch.sqrt( (delta ** 2).mean([1, 2, 3], keepdim=True) )
            max_magnitude = bound_magnitude
            delta = delta * max_magnitude / (avg_magnitude + 1e-4)
        elif self.L12_forced:
            delta = torch.tanh(y_pixels)
            # Divide by 2 to give greater L2 flexibility
            L2_avg_magnitude = torch.sqrt( (delta ** 2).mean([1, 2, 3], keepdim=True) ) / 2
            L1_avg_magnitude = delta.abs().mean([1,2,3], keepdim=True)
            avg_magnitude = torch.min(L1_avg_magnitude, L2_avg_magnitude)
            max_magnitude = bound_magnitude
            delta = delta * max_magnitude / (avg_magnitude + 1e-4)
        elif self.L12_elastic_forced:
            delta = torch.tanh(y_pixels)
            L2_avg_magnitude = torch.sqrt((delta ** 2).mean([1, 2, 3], keepdim=True))
            L1_avg_magnitude = delta.abs().mean([1, 2, 3], keepdim=True)
            avg_magnitude = (L1_avg_magnitude + L2_avg_magnitude) / 2
            max_magnitude = bound_magnitude
            delta = delta * max_magnitude / (avg_magnitude + 1e-4)
        elif self.half:
            delta = torch.tanh(y_pixels)
            l2_diff = (delta ** 2).view(x.size(0), -1).sort(dim=-1)[0]
            median_magnitude = l2_diff.median(1)[0].view(-1, 1, 1, 1)
            # scale down average change if too big.
            max_magnitude = 0.01
            delta = delta * max_magnitude / median_magnitude
        else:
            raise ValueError('Img2Img constraint not specified')
        return delta

    def forward(self, x):
        bound_multiplier = 1
        if self.vary_bound:
            if not self.network or self.network == 'basic':
                # Vary the bound within a batch.  (TODO: add options for a floor + ceiling)
                bound_multiplier = torch.rand(x.size(0), device=x.device)
            else:
                raise RuntimeError('Varying bound currently only implemented for basic net')


        if self.double:
            x = x.repeat((1, 2, 1, 1))  # Repeat along channel dimension

        if self.downsample_to:
            x_orig = x
            x = torch.nn.functional.interpolate(
                x, size=(self.downsample_to, self.downsample_to), mode='bilinear')

        if self.unconditional:
            # Generate noise overlay independent of input image.
            y = torch.zeros_like(x)
        else:
            y = x
        
        # Input to viewmaker is in frequency domain, outputs frequency-domain perturbation.
        if self.spectral:
            # shape still [batch_size, C, W, H]
            y = dct.dct_2d(y)

        if self.network == 'basic' or not self.network:
            y, features = self.basic_net(y, self.num_res_blocks, bound_multiplier)
        elif self.network == 'batch_norm':
            y, features = self.batchnorm_net(y)
        elif self.network == 'small':
            y, features = self.smaller_net(y)
        elif self.network == 'random':
            y = torch.normal(mean=0, std=1, size=y.shape, device=y.device)
        else:
            raise ValueError(f'Network {self.network} not implemented.')

        y_pixels = y[:, :self.num_channels]  # remove displacement field component if extant.

        if self.double:
            y_pixels1, y_pixels2 = y_pixels.chunk(2, dim=1)
            delta1 = self.get_delta(y_pixels1, bound_multiplier)
            delta2 = self.get_delta(y_pixels2, bound_multiplier)
            delta = torch.cat([delta1, delta2], dim=1)
        else:
            delta = self.get_delta(y_pixels, bound_multiplier)

        if self.spectral:
            delta = dct.idct_2d(delta)
        if self.downsample_to:
            x = x_orig
            delta = torch.nn.functional.interpolate(delta, size=x_orig.shape[-2:], mode='bilinear')

        result = x + delta

        if self.clamp:
            result = torch.clamp(result, 0, 1.0)

        if self.affine_layer:
            result = affine.apply_affine(
                result,
                self.affine(features).reshape(-1, 2, 3),
                strength=self.affine_layer,
            )

        if self.color_layer:
            result = affine.apply_color_jitter(
                result,
                self.color(features),
                strength=0.2,
            )

        if self.wb_layer:
            # [batch_size, 1]; [batch_size, 1]
            w, b = torch.chunk(self.wb(features), 2, dim=-1)
            w = w.reshape(-1, 1, 1, 1)
            w = torch.tanh(w) * 0.2 + 1
            b = b.reshape(-1, 1, 1, 1)
            b = torch.tanh(b) * 0.2 
            result = torch.clamp(result * w + b, 0, 1.0)
        
        if self.kernel_layer:
            result = self.apply_kernel(result, self.kernel(features))
        
        if self.warping_layer:
            # [batch_size, H, W, 2]
            warp = y[:, 3:].permute(0, 2, 3, 1)
            result = warping.apply_warp(result, warp, smoothness=0.25, strength=4)

        if self.double:
            result = result.chunk(2, dim=1) # Split up the two views along the channel dimension.

        return result


class ConvLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(ConvLayer, self).__init__()
        reflection_padding = kernel_size // 2
        self.reflection_pad = torch.nn.ReflectionPad2d(reflection_padding)
        self.conv2d = torch.nn.Conv2d(
            in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        out = self.reflection_pad(x)
        out = self.conv2d(out)
        return out


class ResidualBlock(torch.nn.Module):
    """ResidualBlock
    introduced in: https://arxiv.org/abs/1512.03385
    recommended architecture: http://torch.ch/blog/2016/02/04/resnets.html
    """

    def __init__(self, channels, activation='relu'):
        super(ResidualBlock, self).__init__()
        self.conv1 = ConvLayer(channels, channels, kernel_size=3, stride=1)
        self.in1 = torch.nn.InstanceNorm2d(channels, affine=True)
        self.conv2 = ConvLayer(channels, channels, kernel_size=3, stride=1)
        self.in2 = torch.nn.InstanceNorm2d(channels, affine=True)
        self.act = ACTIVATIONS[activation]()

    def forward(self, x):
        residual = x
        out = self.act(self.in1(self.conv1(x)))
        out = self.in2(self.conv2(out))
        out = out + residual
        return out


class UpsampleConvLayer(torch.nn.Module):
    """UpsampleConvLayer
    Upsamples the input and then does a convolution. This method gives better results
    compared to ConvTranspose2d.
    ref: http://distill.pub/2016/deconv-checkerboard/
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, upsample=None):
        super(UpsampleConvLayer, self).__init__()
        self.upsample = upsample
        reflection_padding = kernel_size // 2
        self.reflection_pad = torch.nn.ReflectionPad2d(reflection_padding)
        self.conv2d = torch.nn.Conv2d(
            in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        x_in = x
        if self.upsample:
            x_in = torch.nn.functional.interpolate(
                x_in, mode='nearest', scale_factor=self.upsample)
        out = self.reflection_pad(x_in)
        out = self.conv2d(out)
        return out
