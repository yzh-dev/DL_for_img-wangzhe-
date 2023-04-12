import torch.nn as nn
import torch


# Res18和Res34对应基本残差结构是BasicBlock
class BasicBlock(nn.Module):
    expansion = 1  # 同一个模块中的通道数变化倍数

    def __init__(self, in_channel, out_channel, stride=1, downsample=None, **kwargs):  # downsample=true对应虚线部分
        super(BasicBlock, self).__init__()
        # stride默认等于1，尺寸：output = (input - 3 +2*1)/1 +1 = input
        # 当stride=2时对应虚线部分，尺寸宽高减半
        # pytorch默认向下取整，所以output = (input - 3 +2*1)/2 +1 = input/2 +0.5 = input/2
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel,
                               kernel_size=3, stride=stride, padding=1, bias=False)  # 因为有了BN层，因此不需要使用bias
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel,
                               kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.downsample = downsample

    def forward(self, x):
        # 捷径shortcut的处理
        identity = x
        if self.downsample is not None:  # 下采样的捷径
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = self.relu(out)

        return out


# Res50、Res101、Res152对应的基本残差结构是Bottleneck
class Bottleneck(nn.Module):
    """
    注意：原论文中，在虚线残差结构的主分支上，第一个1x1卷积层的步距是2，第二个3x3卷积层步距是1。
    但在pytorch官方实现过程中是第一个1x1卷积层的步距是1，第二个3x3卷积层步距是2，
    这么做的好处是能够在top1上提升大概0.5%的准确率。
    可参考Resnet v1.5 https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch
    """
    expansion = 4  # 残差块内部的卷积层通道变化倍数

    # in_channel是残差结构中第一个1*1卷积层的通道数
    # out_channel是残差结构中3*3卷积层的通道数
    def __init__(self, in_channel, out_channel, stride=1, downsample=None,
                 groups=1, width_per_group=64):  # ResNet所使用的默认参数groups=1, width_per_group=64
        super(Bottleneck, self).__init__()

        width = int(out_channel * (width_per_group / 64.)) * groups

        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=width,
                               kernel_size=1, stride=1, bias=False)  # squeeze channels
        self.bn1 = nn.BatchNorm2d(width)
        # -----------------------------------------
        self.conv2 = nn.Conv2d(in_channels=width, out_channels=width, groups=groups,
                               kernel_size=3, stride=stride, bias=False, padding=1)
        self.bn2 = nn.BatchNorm2d(width)
        # -----------------------------------------
        self.conv3 = nn.Conv2d(in_channels=width, out_channels=out_channel * self.expansion,
                               kernel_size=1, stride=1, bias=False)  # unsqueeze channels
        self.bn3 = nn.BatchNorm2d(out_channel * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self,
                 block,  # Res18和Res34对应基本残差块是BasicBlock，Res50、Res101、Res152对应的基本残差块是Bottleneck
                 blocks_num,
                 num_classes=1000,
                 include_top=True,  # 基于ResNet搭建更加复杂的网络
                 groups=1,
                 width_per_group=64):
        super(ResNet, self).__init__()
        self.include_top = include_top
        self.in_channel = 64

        self.groups = groups
        self.width_per_group = width_per_group

        self.conv1 = nn.Conv2d(3, self.in_channel, kernel_size=7, stride=2,
                               padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channel)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # 对应网络结构中的第1个残差块conv2_x
        self.layer1 = self._make_layer(block, 64, blocks_num[0])
        # 对应网络结构中的第2个残差块conv3_x
        self.layer2 = self._make_layer(block, 128, blocks_num[1], stride=2)
        # 对应网络结构中的第3个残差块conv4_x
        self.layer3 = self._make_layer(block, 256, blocks_num[2], stride=2)
        # 对应网络结构中的第4个残差块conv5_x
        self.layer4 = self._make_layer(block, 512, blocks_num[3], stride=2)
        if self.include_top:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # output size = (1, 1)
            self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    # first_layer_channel参数指的是任意残差块中第一层的输入通道数，对应主分支上的通道数
    # block_num残差块中对应的残差结构数量
    def _make_layer(self, block, first_layer_channel, block_num, stride=1):
        downsample = None
        # Res50、Res101、Res152残差块的第一个残差结构输入
        if stride != 1 or self.in_channel != first_layer_channel * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channel, first_layer_channel * block.expansion, kernel_size=1, stride=stride,
                          bias=False),
                nn.BatchNorm2d(first_layer_channel * block.expansion))

        layers = []
        # 虚线部分
        layers.append(block(self.in_channel,
                            first_layer_channel,
                            downsample=downsample,
                            stride=stride,
                            groups=self.groups,
                            width_per_group=self.width_per_group))
        self.in_channel = first_layer_channel * block.expansion
        # 剩余的残差结构
        for _ in range(1, block_num):
            layers.append(block(self.in_channel,
                                first_layer_channel,
                                groups=self.groups,
                                width_per_group=self.width_per_group))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        if self.include_top:
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.fc(x)

        return x


def resnet34(num_classes=1000, include_top=True):
    # https://download.pytorch.org/models/resnet34-333f7ec4.pth
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes, include_top=include_top)


def resnet50(num_classes=1000, include_top=True):
    # https://download.pytorch.org/models/resnet50-19c8e357.pth
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes, include_top=include_top)


def resnet101(num_classes=1000, include_top=True):
    # https://download.pytorch.org/models/resnet101-5d3b4d8f.pth
    return ResNet(Bottleneck, [3, 4, 23, 3], num_classes=num_classes, include_top=include_top)


def resnext50_32x4d(num_classes=1000, include_top=True):
    # https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth
    groups = 32
    width_per_group = 4
    return ResNet(Bottleneck, [3, 4, 6, 3],
                  num_classes=num_classes,
                  include_top=include_top,
                  groups=groups,
                  width_per_group=width_per_group)


def resnext101_32x8d(num_classes=1000, include_top=True):
    # https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth
    groups = 32
    width_per_group = 8
    return ResNet(Bottleneck, [3, 4, 23, 3],
                  num_classes=num_classes,
                  include_top=include_top,
                  groups=groups,
                  width_per_group=width_per_group)
