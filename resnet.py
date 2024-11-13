import torch
import torch.nn as nn
import torch.nn.functional as F

# Custom Conv2d class as specified
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
        std = weight.view(weight.size(0), -1).std(dim=1).view(-1, 1, 1, 1) + 1e-5
        weight = weight / std.expand_as(weight)
        return F.conv2d(x, weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)

# Basic block with Conv2d and GroupNorm instead of BatchNorm
class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, num_groups=16):
        super(BasicBlock, self).__init__()
        self.conv1 = Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.gn1 = nn.GroupNorm(num_groups, out_channels)
        self.conv2 = Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.gn2 = nn.GroupNorm(num_groups, out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.GroupNorm(num_groups, out_channels)
            )
            
    def forward(self, x):
        out = F.relu(self.gn1(self.conv1(x)))
        out = self.gn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out)

# ResNet-20 model definition with GroupNorm
class ResNet20(nn.Module):
    def __init__(self, num_classes=10, num_groups=16):
        super(ResNet20, self).__init__()
        self.in_channels = 16
        
        # Initial convolutional layer with GroupNorm
        self.conv1 = Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.gn1 = nn.GroupNorm(num_groups, 16)
        
        # ResNet layers with BasicBlock
        self.layer1 = self._make_layer(16, 3, stride=1, num_groups=num_groups)
        self.layer2 = self._make_layer(32, 3, stride=2, num_groups=num_groups)
        self.layer3 = self._make_layer(64, 3, stride=2, num_groups=num_groups)
        
        # Fully connected layer
        self.fc = nn.Linear(64, num_classes)

    def _make_layer(self, out_channels, blocks, stride, num_groups):
        layers = []
        layers.append(BasicBlock(self.in_channels, out_channels, stride, num_groups=num_groups))
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(BasicBlock(out_channels, out_channels, stride=1, num_groups=num_groups))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.gn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)
        return self.fc(out)
