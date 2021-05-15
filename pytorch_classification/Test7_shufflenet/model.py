from typing import List, Callable

import torch
from torch import Tensor
import torch.nn as nn


#! shuffle操作
def channel_shuffle(x: Tensor, groups: int) -> Tensor:

    batch_size, num_channels, height, width = x.size() #& tensorflow中tensor的排列是：batch，channel，h，w
    channels_per_group = num_channels // groups        # 将channel划分为group组

    #& reshape 
    # * [batch_size, num_channels, height, width] -> [batch_size, groups, channels_per_group, height, width]
    x = x.view(batch_size, groups, channels_per_group, height, width)

    #& transpose
    x = torch.transpose(x, 1, 2).contiguous()   # 将维度1（groups）与维度2（channels_per_group）的信息进行调换
                                                # .contiguous() 将tensor数据转化为内存中存储为连续的数据

    #& flatten
    x = x.view(batch_size, -1, height, width)   # 然后再还原为tensor的形式； -1表示可以通过计算自动填充

    return x


class InvertedResidual(nn.Module):
    def __init__(self, input_c: int, output_c: int, stride: int):
        super(InvertedResidual, self).__init__()

        if stride not in [1, 2]:                            #  stride只能取1火折子2
            raise ValueError("illegal stride value.")
        self.stride = stride

        assert output_c % 2 == 0                            # output必须是2的整数倍，因为下面的block要对其一分为2 
        branch_features = output_c // 2                     # 两个分支的channel都是一样的

        # 当stride为1时，input_channel应该是branch_features的两倍
        #* python中 '<<' 是位运算，可理解为计算×2的快速方法
        assert (self.stride != 1) or (input_c == branch_features << 1)
            # 如果stride=2，那么可以继续往下走
            # 如果stride=1，那么对应图c，输入和输出的特征矩阵的channel要保持一致（满足G1准则）
            # input_c == branch_features << 1 等价于 input_c == branch_features*2

        #^ 先定义好左边的捷径分支branch1
        if self.stride == 2:                                # 对应图d
            #^ branch1是左边的捷径分支，首先3*3的DW卷积（s=2）+BN，然后1*1普通卷积+BN+ReLU
            self.branch1 = nn.Sequential(
                self.depthwise_conv(input_c, input_c, kernel_s=3, stride=self.stride, padding=1),
                    # 输入和输出的channel都是一样的，可以看出就是DW卷积
                nn.BatchNorm2d(input_c),            # DW卷积后的BN层，BN的输入等于DW卷积的输出
                nn.Conv2d(input_c, branch_features, kernel_size=1, stride=1, padding=0, bias=False),
                    # 普通的1*1卷积，输出的channel就是上面计算的branch_features
                nn.BatchNorm2d(branch_features),    # 有BN层，上面的conv的bais不需要
                nn.ReLU(inplace=True)               # 接上BN层
            )
        else:
            self.branch1 = nn.Sequential()
            #^ 图c的左边分支branch1没有做任何处理，所以就是一个空的Sequential()

        #^ 再定义右边的主干分支branch2，图c和图d
        #^ 右边的结构是：首先是1*1普通卷积+BN+ReLU，然后是3*3的DW卷积（stride为1或2）+BN，最后是1*1普通卷积+BN+ReLU
        self.branch2 = nn.Sequential(
            nn.Conv2d(input_c if self.stride > 1 else branch_features, branch_features, kernel_size=1,
                      stride=1, padding=0, bias=False),
                # 先看输入的channel：
                    # 如果stride=1，对应图c，左、右分支各一半，输入的channel等于input_c/2，即branch_features
                    # 如果stride=2，对应图d，左、右分支的输入的channel都等于input_c
                # 再看输出的channel：
                    # 如果stride=1，对应图c，是，输出的channel都是input_c/2
                    #? 如果stride=2，对应图d，对于每一个stage的第一个block而言，输出channel是要进行翻倍的???
            nn.BatchNorm2d(branch_features),
            nn.ReLU(inplace=True),
            self.depthwise_conv(branch_features, branch_features, kernel_s=3, stride=self.stride, padding=1),
                #* DW卷积的输入和输出是保持不变的
                #* stride=self.stride，可以为1或者2，这就是图c和图d的不同之处
            nn.BatchNorm2d(branch_features),
            nn.Conv2d(branch_features, branch_features, kernel_size=1, stride=1, padding=0, bias=False),
                # 最后的1*1普通卷积，输入和输出的channel都是一样的
            nn.BatchNorm2d(branch_features),
            nn.ReLU(inplace=True)
        )

    @staticmethod
    def depthwise_conv(input_c: int,            #^ DW卷积是逐层卷积，是分组卷积中g=c的特殊情况
                       output_c: int,           #^ DW卷积的输入和输出的channel是一样的，写法和普通的卷积一样
                       kernel_s: int,           
                       stride: int = 1,
                       padding: int = 0,
                       bias: bool = False) -> nn.Conv2d:    #^ 因为DW卷积后有BN层，所以不用bias
        return nn.Conv2d(in_channels=input_c, out_channels=output_c, kernel_size=kernel_s,
                         stride=stride, padding=padding, bias=bias, groups=input_c)

    def forward(self, x: Tensor) -> Tensor:
        #^ 图c：stride == 1
        if self.stride == 1:                
            x1, x2 = x.chunk(2, dim=1)      #* x.chunk均分处理，分为2份；tensor中channel的维度号是1（b,C,H,W）
            out = torch.cat((x1, self.branch2(x2)), dim=1)  
                # 左边的捷径分支不做任何处理，就是x1；右边主干分支传入branch2
                #* concat拼接，按照维度进行拼接； tensor中channel的维度号是1（b,C,H,W）
        #^ 图d：stride == 2
        else:
            out = torch.cat((self.branch1(x), self.branch2(x)), dim=1)
                # 左右分支分别通过branch1和2，然后按照channel进行concat

        out = channel_shuffle(out, 2)   #* 最后输出进行channel_shuffle操作

        return out


class ShuffleNetV2(nn.Module):
    def __init__(self,
                 stages_repeats: List[int],
                 stages_out_channels: List[int],
                 num_classes: int = 1000,
                 inverted_residual: Callable[..., nn.Module] = InvertedResidual):
        super(ShuffleNetV2, self).__init__()

        if len(stages_repeats) != 3:        #* 只有3个stage：stage2、3、4
            raise ValueError("expected stages_repeats as list of 3 positive ints")
        if len(stages_out_channels) != 5:   #* 共有5个卷积部分：Conv1，stage2、3、4，Conv5
            raise ValueError("expected stages_out_channels as list of 5 positive ints")
        self._stage_out_channels = stages_out_channels

        #& 定义 conv1
        # input RGB image
        input_channels = 3                  # 输入的是RGB彩色图片，channel=3
        output_channels = self._stage_out_channels[0]   # 第一个卷积部分Conv1的输出是列表[0]
            # 以此构建conv1

        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True)
        )
        input_channels = output_channels        # 第一个卷积的输出是第二个卷积的输入，channel进行赋值；定义MaxPool

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Static annotations for mypy   
        self.stage2: nn.Sequential      #* 申明：stage2、3、4都是通过 nn.Sequential 实现的
        self.stage3: nn.Sequential
        self.stage4: nn.Sequential

        #& 定义 stage2、3、4
        stage_names = ["stage{}".format(i) for i in [2, 3, 4]]          # 构建stage_names
        for name, repeats, output_channels in zip(stage_names, stages_repeats,
                                                  self._stage_out_channels[1:]):
                # stage_names, stages_repeats都只有3个元素，stage_out_channels也只会取3个元素，zip结束后就不会继续往后
            seq = [inverted_residual(input_channels, output_channels, 2)]   
                    #* 先定义第一个block，stride=2
            for i in range(repeats - 1):    # 再遍历剩下的block     
                seq.append(inverted_residual(output_channels, output_channels, 1))
                    #* 依次添加剩下的block，它们的stride=1，且输入和输出的channel一样
            setattr(self, name, nn.Sequential(*seq))    #! 设置变量，并传入值
            input_channels = output_channels            # 当前的输出channel赋值给下一层的输入channel
                # stage2的输出是stage3的输入；stage3的输出是stage4的输入；stage3的输出是conv5的输入

        #& 定义conv5
        output_channels = self._stage_out_channels[-1]   # 列表的最后一个元素，就是conv5的输出
        self.conv5 = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True)
        )

        #& 定义 fc
        self.fc = nn.Linear(output_channels, num_classes)   # fc的输出就是num_classes

    def _forward_impl(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.conv5(x)
        x = x.mean([2, 3])  #! global pool，直接使用mean方法做全局池化
            #! [2, 3] 这两个对应的是h和w维度，mean之后这两个维度就没有了，只有batch和channel维度
        x = self.fc(x)
        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)



#& 想实现不同的版本，不同的输入列表就可以了
def shufflenet_v2_x1_0(num_classes=1000):
    """
    Constructs a ShuffleNetV2 with 1.0x output channels, as described in
    `"ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design"
    <https://arxiv.org/abs/1807.11164>`.
    weight: https://download.pytorch.org/models/shufflenetv2_x1-5666bf0f80.pth

    :param num_classes:
    :return:
    """
    # 这里对应的是表格中的1*版本
    model = ShuffleNetV2(stages_repeats=[4, 8, 4],          #* 每个stage中的block的重复次数列表
                         stages_out_channels=[24, 116, 232, 464, 1024],
                            #* 共有5个卷积部分：Conv1，stage2、3、4，Conv5；每个卷积部分的输出的channel
                         num_classes=num_classes)

    return model


def shufflenet_v2_x0_5(num_classes=1000):
    """
    Constructs a ShuffleNetV2 with 0.5x output channels, as described in
    `"ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design"
    <https://arxiv.org/abs/1807.11164>`.
    weight: https://download.pytorch.org/models/shufflenetv2_x0.5-f707e7126e.pth

    :param num_classes:
    :return:
    """
    model = ShuffleNetV2(stages_repeats=[4, 8, 4],
                         stages_out_channels=[24, 48, 96, 192, 1024],
                         num_classes=num_classes)

    return model
