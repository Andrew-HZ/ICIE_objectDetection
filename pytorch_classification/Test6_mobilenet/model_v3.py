from typing import Callable, List, Optional

import torch
from torch import nn, Tensor
from torch.nn import functional as F
from functools import partial


def _make_divisible(ch, divisor=8, min_ch=None):
    # * 将channel调整到离divisor的整数倍最近的数字；这样对硬件更好，也有利于硬件的提升
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if min_ch is None:
        min_ch = divisor
    new_ch = max(min_ch, int(ch + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_ch < 0.9 * ch:
        new_ch += divisor
    return new_ch


class ConvBNActivation(nn.Sequential):
    def __init__(self,
                 in_planes: int,
                 out_planes: int,
                 kernel_size: int = 3,
                 stride: int = 1,
                 groups: int = 1,
                 norm_layer: Optional[Callable[..., nn.Module]] = None,             # 对应卷积后的BN层
                 activation_layer: Optional[Callable[..., nn.Module]] = None):      # 对应激活函数
        padding = (kernel_size - 1) // 2
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d                     # 默认使用BN层        
        if activation_layer is None:
            activation_layer = nn.ReLU6                     # 默认使用ReLU6激活函数
        super(ConvBNActivation, self).__init__(nn.Conv2d(in_channels=in_planes,
                                                         out_channels=out_planes,
                                                         kernel_size=kernel_size,
                                                         stride=stride,
                                                         padding=padding,
                                                         groups=groups,
                                                         bias=False),               # * 由于使用了BN层，所以这里不用bias
                                               norm_layer(out_planes),              # * 卷积后面使用BN
                                               activation_layer(inplace=True))      # * BN层后面使用激活函数


# ! SE模块：注意力机制模块
# ^ SE模块包括两个fc层：第一个fc层的节点个数等于输入的channel除以4，其激活函数是ReLU
# ^                   第二个fc层的节点个数和输入的channel保持一致，其激活函数是hard sigmoid：
class SqueezeExcitation(nn.Module):
    def __init__(self, input_c: int, squeeze_factor: int = 4):  # squeeze_factor压缩比例，这里取压缩4倍
        super(SqueezeExcitation, self).__init__()
        squeeze_c = _make_divisible(input_c // squeeze_factor, 8)   # 计算第一个fc层的节点个数
            # 第一个fc层的节点个数等于输入的channel除以4，并调整到最近的8的整数倍
        self.fc1 = nn.Conv2d(input_c, squeeze_c, 1)     # * 这里直接用1*1的卷积来表示全连接层
        self.fc2 = nn.Conv2d(squeeze_c, input_c, 1)     # * 第二个fc层的节点个数（输出的channel）和输入的channel保持一致

    def forward(self, x: Tensor) -> Tensor:         
        # * 传入的x是一个特征矩阵（x: Tensor）；该正向传播返回的也是一个特征矩阵（-> Tensor）
        # * 需要对每一个channel都进行池化操作
        scale = F.adaptive_avg_pool2d(x, output_size=(1, 1))    
            # * 自适应的池化操作，输出的维度是1*1；这样就可以将特征矩阵的每一个channel的数据平均池化为一个数字（1*1大小）
        scale = self.fc1(scale)
        scale = F.relu(scale, inplace=True)             # * 分别经过fc1和其对应的relu激活
        scale = self.fc2(scale)
        scale = F.hardsigmoid(scale, inplace=True)      # * 分别经过fc2和其对应的hardsigmoid激活
        return scale * x                         # ^ 得到的数要与channel上的数字进行相乘


# ! 这里的InvertedResidualConfig对应的是M哦bileNetV3中的每一个bneck结构的参数配置
class InvertedResidualConfig:
    def __init__(self,
                 input_c: int,              # 输入的特征矩阵大小，这里主要关注其channel
                 kernel: int,               # kernel size
                 expanded_c: int,           # * expanded_c表示第一个升维的1*1卷积层的卷积核个数
                 out_c: int,                
                 use_se: bool,              # 是否使用注意力机制
                 activation: str,           # 激活函数：HS表示hard-swish激活函数，RE表示ReLU;
                 stride: int,
                 width_multi: float):       # 类似于阿尔法参数，调整每个卷积层的channel
        self.input_c = self.adjust_channels(input_c, width_multi)       # 首先调整出输入的channel
        self.kernel = kernel
        self.expanded_c = self.adjust_channels(expanded_c, width_multi)
        self.out_c = self.adjust_channels(out_c, width_multi)
        self.use_se = use_se
        self.use_hs = activation == "HS"  # whether using h-swish activation
        self.stride = stride

    @staticmethod               # 静态方法
    def adjust_channels(channels: int, width_multi: float):     # 调整channel方法
        return _make_divisible(channels * width_multi, 8)       #  channels * 因子，然后调整到最近的8的整数倍


# ! V3的倒残差结构 （整个结构）
class InvertedResidual(nn.Module):
    def __init__(self,
                 cnf: InvertedResidualConfig,                   # 这就是上面的cnf文件
                 norm_layer: Callable[..., nn.Module]):
        super(InvertedResidual, self).__init__()

        if cnf.stride not in [1, 2]:                            # stride只能是1或者2，否则就非法
            raise ValueError("illegal stride value.")

        self.use_res_connect = (cnf.stride == 1 and cnf.input_c == cnf.out_c)   # 是否使用short-cut连接
            # * 要求stride=1（不改变h和w），且输入和输出的shape一样（即保证channel一样即可），这样才能相加

        layers: List[nn.Module] = []        # & 学会这招 包含注释信息的定义
            # * 定义一个空列表，其元素是nn.Module类型   写为: List[nn.Module]

        activation_layer = nn.Hardswish if cnf.use_hs else nn.ReLU  # 两个激活函数的选择
            # pytorch1.7以上才有Hardswish方法

        # & expand结构 （不一定有）
        # * 第一个bncek结构，input channel和expanded channel是相等的，所以该结构没有1*1的卷积层升维；
        # * 其他的bneck结构，二者不想等，所以都有1*1的卷积层升维；
        if cnf.expanded_c != cnf.input_c:                       # ^ 如果expanded和input不相等，就有1*1的卷积层
            layers.append(ConvBNActivation(cnf.input_c,
                                           cnf.expanded_c,
                                           kernel_size=1,       # 1*1卷积层
                                           norm_layer=norm_layer,
                                           activation_layer=activation_layer))

        # & depthwise DW卷积层
        layers.append(ConvBNActivation(cnf.expanded_c,                  # *上一层输出的特征矩阵channel，作为本层的输入
                                       cnf.expanded_c,                  # * DW卷积的输入和输出的channel是保持一致的
                                       kernel_size=cnf.kernel,          # 相关参数和配置文件保持一致
                                       stride=cnf.stride,           
                                       groups=cnf.expanded_c,           
                                            # * DW卷积，对每一个channel都单独使用channel=1的卷积核处理；所以groups=channel
                                       norm_layer=norm_layer,
                                       activation_layer=activation_layer))

        if cnf.use_se:
            layers.append(SqueezeExcitation(cnf.expanded_c))            # 如果需要SE模块，就添加进去

        # & project 最后的1*1降维结构
        layers.append(ConvBNActivation(cnf.expanded_c,                  # * 上一层输出的特征矩阵channel，作为本层的输入            
                                       cnf.out_c,
                                       kernel_size=1,                   # 1*1卷积
                                       norm_layer=norm_layer,
                                       activation_layer=nn.Identity))   # * nn.Identity 线性激活，不需要做任何处理
                                            # 可以点进去看一下，forward函数中 return input，说明没有做任何处理

        self.block = nn.Sequential(*layers)     # ^ 将网络层列表变为网络
        self.out_channels = cnf.out_c
        self.is_strided = cnf.stride > 1

    def forward(self, x: Tensor) -> Tensor:
        result = self.block(x)
        if self.use_res_connect:
            result += x             # 如果当前层有捷径分支，就相加

        return result


class MobileNetV3(nn.Module):
    def __init__(self,
                 inverted_residual_setting: List[InvertedResidualConfig],       # * 一系列bneck结构参数列表（倒残差结构）
                 last_channel: int,                                             # 倒数第2个1*1卷积层（全连接层）的节点个数
                 num_classes: int = 1000,
                 block: Optional[Callable[..., nn.Module]] = None,              # block是倒残差模块，默认为None
                 norm_layer: Optional[Callable[..., nn.Module]] = None):
        super(MobileNetV3, self).__init__()

        # 要求传入inverted_residual_setting，且是列表，且会遍历列表，判断都是cnf文件
        if not inverted_residual_setting:
            raise ValueError("The inverted_residual_setting should not be empty.")
        elif not (isinstance(inverted_residual_setting, List) and
                  all([isinstance(s, InvertedResidualConfig) for s in inverted_residual_setting])):
            raise TypeError("The inverted_residual_setting should be List[InvertedResidualConfig]")

        if block is None:
            block = InvertedResidual                                            # block是倒残差模块

        if norm_layer is None:
            norm_layer = partial(nn.BatchNorm2d, eps=0.001, momentum=0.01)      # norm_layer设置为BN
                        # * partial是python的语法，传入两个默认参数

        layers: List[nn.Module] = []

        # building first layer
        firstconv_output_c = inverted_residual_setting[0].input_c       
                # 获取第一个bncek结构的输入channel，即网络中第一个卷积层的输出channel
            # 下面的卷积层就是网络中第一个卷积层conv2d
        layers.append(ConvBNActivation(3,                               # 输入的是RGB图片，channel=3            
                                       firstconv_output_c,              # 输出的channel是第一个bncek结构的输入channel
                                       kernel_size=3,                   # 3*3卷积
                                       stride=2,
                                       norm_layer=norm_layer,
                                       activation_layer=nn.Hardswish))

        # building inverted residual blocks
        for cnf in inverted_residual_setting:           # 遍历，传入cnf
            layers.append(block(cnf, norm_layer))

        # building last several layers
        lastconv_input_c = inverted_residual_setting[-1].out_c          # 最后一个bneck结构的输出就是倒数第二个卷积层lastconv的输入
        lastconv_output_c = 6 * lastconv_input_c                        # * 输出的channel是输入的6倍；看图，输入160，输出960
        layers.append(ConvBNActivation(lastconv_input_c,
                                       lastconv_output_c,
                                       kernel_size=1,                   # 1*1卷积
                                       norm_layer=norm_layer,
                                       activation_layer=nn.Hardswish))
        
        # ^ 这个features就是提取特征的主干网络
        self.features = nn.Sequential(*layers)                          

        self.avgpool = nn.AdaptiveAvgPool2d(1)                          # 使用自适应的平均池化
        
        # ^ 分类网络就是两个fc层                                        
        self.classifier = nn.Sequential(nn.Linear(lastconv_output_c, last_channel), # 输入和上面的输出一致，输出和最后的channel一致
                                        nn.Hardswish(inplace=True),
                                        nn.Dropout(p=0.2, inplace=True),            # * 先激活，再drop out    
                                        nn.Linear(last_channel, num_classes))       # 输入是last_channel，输出是类别数目

        # initial weights 初始化权重
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def _forward_impl(self, x: Tensor) -> Tensor:
            # ^ 提取特征网络features + 平均池化 + 展平 + 分类
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)

        return x

    def forward(self, x: Tensor) -> Tensor:     # 输入的x是打包好的batch图片
        return self._forward_impl(x)


def mobilenet_v3_large(num_classes: int = 1000,
                       reduced_tail: bool = False) -> MobileNetV3:
    """
    Constructs a large MobileNetV3 architecture from
    "Searching for MobileNetV3" <https://arxiv.org/abs/1905.02244>.

    weights_link:  已训练好的权重
    https://download.pytorch.org/models/mobilenet_v3_large-8738ca79.pth

    Args:
        num_classes (int): number of classes
        reduced_tail (bool): If True, reduces the channel counts of all feature layers
            between C4 and C5 by 2. It is used to reduce the channel redundancy in the
            backbone for Detection and Segmentation.
    """
    width_multi = 1.0           # 阿尔法参数，针对channel的
    bneck_conf = partial(InvertedResidualConfig, width_multi=width_multi)       # 传入默认参数
    adjust_channels = partial(InvertedResidualConfig.adjust_channels, width_multi=width_multi)

    reduce_divider = 2 if reduced_tail else 1           # 若为1，则和原论文一致；若为2，则网络模型更小

    inverted_residual_setting = [
        # input_c, kernel, expanded_c, out_c, use_se, activation, stride
        bneck_conf(16, 3, 16, 16, False, "RE", 1),
        bneck_conf(16, 3, 64, 24, False, "RE", 2),  # C1
        bneck_conf(24, 3, 72, 24, False, "RE", 1),
        bneck_conf(24, 5, 72, 40, True, "RE", 2),  # C2
        bneck_conf(40, 5, 120, 40, True, "RE", 1),
        bneck_conf(40, 5, 120, 40, True, "RE", 1),
        bneck_conf(40, 3, 240, 80, False, "HS", 2),  # C3
        bneck_conf(80, 3, 200, 80, False, "HS", 1),
        bneck_conf(80, 3, 184, 80, False, "HS", 1),
        bneck_conf(80, 3, 184, 80, False, "HS", 1),
        bneck_conf(80, 3, 480, 112, True, "HS", 1),
        bneck_conf(112, 3, 672, 112, True, "HS", 1),
        bneck_conf(112, 5, 672, 160 // reduce_divider, True, "HS", 2),  # C4
        bneck_conf(160 // reduce_divider, 5, 960 // reduce_divider, 160 // reduce_divider, True, "HS", 1),
        bneck_conf(160 // reduce_divider, 5, 960 // reduce_divider, 160 // reduce_divider, True, "HS", 1),
    ]
    last_channel = adjust_channels(1280 // reduce_divider)  # C5

    return MobileNetV3(inverted_residual_setting=inverted_residual_setting,
                       last_channel=last_channel,
                       num_classes=num_classes)


def mobilenet_v3_small(num_classes: int = 1000,
                       reduced_tail: bool = False) -> MobileNetV3:
    """
    Constructs a large MobileNetV3 architecture from
    "Searching for MobileNetV3" <https://arxiv.org/abs/1905.02244>.
    Args:
        num_classes (int): number of classes
        reduced_tail (bool): If True, reduces the channel counts of all feature layers
            between C4 and C5 by 2. It is used to reduce the channel redundancy in the
            backbone for Detection and Segmentation.
    """
    width_multi = 1.0
    bneck_conf = partial(InvertedResidualConfig, width_multi=width_multi)
    adjust_channels = partial(InvertedResidualConfig.adjust_channels, width_multi=width_multi)

    reduce_divider = 2 if reduced_tail else 1

    inverted_residual_setting = [
        # input_c, kernel, expanded_c, out_c, use_se, activation, stride
        bneck_conf(16, 3, 16, 16, True, "RE", 2),  # C1
        bneck_conf(16, 3, 72, 24, False, "RE", 2),  # C2
        bneck_conf(24, 3, 88, 24, False, "RE", 1),
        bneck_conf(24, 5, 96, 40, True, "HS", 2),  # C3
        bneck_conf(40, 5, 240, 40, True, "HS", 1),
        bneck_conf(40, 5, 240, 40, True, "HS", 1),
        bneck_conf(40, 5, 120, 48, True, "HS", 1),
        bneck_conf(48, 5, 144, 48, True, "HS", 1),
        bneck_conf(48, 5, 288, 96 // reduce_divider, True, "HS", 2),  # C4
        bneck_conf(96 // reduce_divider, 5, 576 // reduce_divider, 96 // reduce_divider, True, "HS", 1),
        bneck_conf(96 // reduce_divider, 5, 576 // reduce_divider, 96 // reduce_divider, True, "HS", 1)
    ]
    last_channel = adjust_channels(1024 // reduce_divider)  # C5

    return MobileNetV3(inverted_residual_setting=inverted_residual_setting,
                       last_channel=last_channel,
                       num_classes=num_classes)
