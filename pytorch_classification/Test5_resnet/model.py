import torch.nn as nn
import torch


# * 对应18层和34层的残差结构
class BasicBlock(nn.Module):                            
    expansion = 1        # * 表示残差结构的主分支中，卷积核的个数是否发生变化

    def __init__(self, in_channel, out_channel, stride=1, downsample=None, **kwargs):
                        # * downsample=None 表示是否有虚线的那个残差结构（分支上是否有1*1的卷积进行降维）
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel,
                               kernel_size=3, stride=stride, padding=1, bias=False)
                        # ^ 实线的残差结构，分支上stride=1； 虚线的残差结构，分支上的stride=2，且有1*1卷积
                        # * 每新的一组卷积单元的第一个卷积层，都有下采样操作，对应的都是虚线的残差结构
                        # * stride=1的时候：output = (input -3 +2*1)/1 +1 = input，输入和输出是一样的
                        # * stride=2的时候：output = (input -3 +2*1)/2 +1 = input/2 +0.5 = input/2 （向下取整）
                        # & 由于有了BN层，所以卷积层都不使用bias，bias=False
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel,
                               kernel_size=3, stride=1, padding=1, bias=False)
                        # * 无论是实线还是虚线对应的残差结构，主分支的第二个卷积层都有stride=1
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.downsample = downsample        # 下采样方法，默认为None

    def forward(self, x):
        identity = x                                # shortcut上的分支值
        if self.downsample is not None:
            identity = self.downsample(x)           # 如果downsample不为None，则求出分支上的输出

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)                         # * 注意：主分支上的第二个卷积层是没有激活函数的

        out += identity
        out = self.relu(out)                        # * 两个分支的结果相加后，再经过relu激活

        return out


# * 对应50层、101层、152层的残差结构
class Bottleneck(nn.Module):
    """
    注意：原论文中，在虚线残差结构的主分支上，第一个1x1卷积层的步距是2，第二个3x3卷积层步距是1。
    但在pytorch官方实现过程中是第一个1x1卷积层的步距是1，第二个3x3卷积层步距是2，
    这么做的好处是能够在top1上提升大概0.5%的准确率。
    可参考Resnet v1.5 https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch
    """
    expansion = 4         # * 第三层卷积层的深度是前两层卷积层的4倍，所以expansion = 4

    def __init__(self, in_channel, out_channel, stride=1, downsample=None,
                 groups=1, width_per_group=64):                 
                # 多传入的两个参数是针对ResNetXt的：groups是C， width_per_group是d（C=32，d=4）
        super(Bottleneck, self).__init__()

        width = int(out_channel * (width_per_group / 64.)) * groups    
                # 如果不用组卷积，width = out_channel
                # * 使用C=32，d=4这组参数， out*4/64*32 = out*2，正好说明了ResNetXt的输入是RestNet输入的两倍

        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=width,
                               kernel_size=1, stride=1, bias=False)  # squeeze channels
                            # * 无论是实线还是虚线，stride=1，即output = (input -3 +2*1)/1 +1 = input，输入和输出是一样的
        self.bn1 = nn.BatchNorm2d(width)
        # -----------------------------------------
        self.conv2 = nn.Conv2d(in_channels=width, out_channels=width, groups=groups,
                               kernel_size=3, stride=stride, bias=False, padding=1)
                            # * 第二层卷积：实现对应的stride=1；虚线对应的stride=2；所以这里是一个传入的参数stride=stride
                            # * 第二层卷积：输入和输出都等于out_channels（这里优化为width）
        self.bn2 = nn.BatchNorm2d(width)
        # -----------------------------------------
        self.conv3 = nn.Conv2d(in_channels=width, out_channels=out_channel*self.expansion,
                               kernel_size=1, stride=1, bias=False)  # unsqueeze channels
                            # * 第三层卷积：输出的卷积核深度 是 输入的卷积核深度的4倍 (expansion = 4)
        self.bn3 = nn.BatchNorm2d(out_channel*self.expansion)
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
                 block,                         # 这里的block对应的就是残差结构：BasicBlock或者Bottleneck
                 blocks_num,                    # 使用的残差结果的数目，这是一个列表参数
                 num_classes=1000,
                 include_top=True,              # 可以在这个基础上搭建更复杂的网络，默认为True
                 groups=1,
                 width_per_group=64):
        super(ResNet, self).__init__()
        self.include_top = include_top
        self.in_channel = 64

        self.groups = groups
        self.width_per_group = width_per_group

        self.conv1 = nn.Conv2d(3, self.in_channel, kernel_size=7, stride=2,
                               padding=3, bias=False)                               # 第一层卷积是7*7的，stride=2；输入的维度是3 RGB
        self.bn1 = nn.BatchNorm2d(self.in_channel)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # * self.layer 1 2 3 4对应的就是表格中的conv2 3 4 5_x这样的残差结构
        # * 64 128 256 512 对应的都是每一组残差结构中  第一个卷积层的个数
        self.layer1 = self._make_layer(block, 64, blocks_num[0])             # 默认 stride=1
        self.layer2 = self._make_layer(block, 128, blocks_num[1], stride=2)
        self.layer3 = self._make_layer(block, 256, blocks_num[2], stride=2)
        self.layer4 = self._make_layer(block, 512, blocks_num[3], stride=2)

        if self.include_top:                             # 默认为True
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # output size = (1, 1)
                    # * 使用自适应的平均池化下采样操作，输出的特征矩阵的h和w都是1
            self.fc = nn.Linear(512 * block.expansion, num_classes)   
                    # * 然后进行展平操作，由于特征图是1*1*channel，所以展平就是channel
                    # * 对于18层、34层：最后展平后的深度是512，expansion=1，可写为512 * block.expansion
                    # * 对应50层、101层、152层：最后展平后的深度是2048，其expansion=4，即深度为512 * block.expansion

        # * 对卷积层进行初始化操作
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def _make_layer(self, block, channel, block_num, stride=1):             # 默认 stride=1
            # * block有两种：BasicBlock（对应18层、34层）、Bottleneck（对应50层、101层、152层）
            # * channel对应的是残差结构中第一层卷积层的卷积核个数：对于BasicBlock（第一层和第二层的卷积核个数一样）；对于Bottleneck（第三层的卷积核个数是第一层的卷积核个数的4倍）
            # * block_num表示每一组残差结构中包含多少个残差结构，即重复了多少次

        # ! 下面看到底有没有下采样？
        downsample = None
        if stride != 1 or self.in_channel != channel * block.expansion:
            # * stride=1：3种情况：BasicBlock（对应18层、34层）实线的时候；Bottleneck（对应50层、101层、152层）实线+虚线的时候
            # * stride !=1：仅1种情况：BasicBlock（对应18层、34层）虚线的时候 （第一层）
            # * in_channel = channel * block.expansion： 3种情况：BasicBlock：expansion=1 实线+虚线的时候；Bottleneck情况较复杂
            # ^ 对于Bottleneck（对应50层、101层、152层）的第一层卷积层（虚线）：
                    # ^ 对于conv2_X，只调整特征矩阵的深度，高度和宽度不用调整；
                    # ^ 而conv3_X，conv4_X，conv5_X中的第一个残差结构，不仅调整特征矩阵的深度，高度和宽度还要缩小为原来的一半。
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channel, channel * block.expansion, kernel_size=1, stride=stride, bias=False),
                    # 输入的深度是self.in_channel；输出的深度是channel * block.expansion；
                    # ! 对于conv2 3 4 5中重复单元内的第一组残差结构中，深度都会扩展为原来的4倍；以后不再改变深度
                    # * 捷径分支上都是1*1的卷积核，个数确实是输入的4倍数（expansion）
                    # ! stride=1：时并不影响h和w（对应Conv2）;stride=2：h和w会缩小为原来的一半（对应conv3 4 5）
                nn.BatchNorm2d(channel * block.expansion))  
                    # 所以BN层传入的深度也是channel * block.expansion

        layers = []    # 先定义一个空列表
 
        # * 先存入第一层残差结构（包含虚线的残差结构）  （残差结构重复了block_num次）
        layers.append(block(self.in_channel,                            # 输入的深度 64
                            channel,                                    # 主分支的第一个卷积层的卷积核个数
                            downsample=downsample,                      # 18和34层downsample=None；50、101和152层，深度翻4倍，h和w不变
                            stride=stride,
                            groups=self.groups,
                            width_per_group=self.width_per_group))
        self.in_channel = channel * block.expansion

        # * 再存入剩下的残差结构（实线的），for loop
        for _ in range(1, block_num):                   # 第0层已经搭建好了
            layers.append(block(self.in_channel,        # 传入输入的深度
                                channel,                # 第一层卷积的卷积核个数 （后面的卷积核个数一样，不会再改变深度，所以输出也是这个）
                                groups=self.groups,
                                width_per_group=self.width_per_group))

        # & 构建好列表layer后，再根据非关键字参数的形式传给nn.Sequential
        return nn.Sequential(*layers)                   # 将刚刚的层结构组合在一起

    # * 正向传播过程
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
    # 如果是18层的，[2, 2, 2, 2]

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
                  width_per_group=width_per_group)    # 需要多传入这两个参数


def resnext101_32x8d(num_classes=1000, include_top=True):
    # https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth
    groups = 32
    width_per_group = 8
    return ResNet(Bottleneck, [3, 4, 23, 3],
                  num_classes=num_classes,
                  include_top=include_top,
                  groups=groups,
                  width_per_group=width_per_group)
