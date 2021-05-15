from torch import nn
import torch


def _make_divisible(ch, divisor=8, min_ch=None):
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

# & 基本模块CBR结构：
class ConvBNReLU(nn.Sequential):
    def __init__(self, in_channel, out_channel, kernel_size=3, stride=1, groups=1):     
                # * 如果group=in_channel，那么就是DW卷积
        padding = (kernel_size - 1) // 2            # 填充参数
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_channel, out_channel, kernel_size, stride, padding, groups=groups, bias=False),  # kernel_size卷积核的大小
            nn.BatchNorm2d(out_channel),            
                # * 由于有BN层，所以不用偏置，bias=False； BN层的输入就是上一层的输出out_channel
            nn.ReLU6(inplace=True)                  
                # * 再次记住：先BN，然后激活
        )


# & 倒残差结构：两头细 中间粗
class InvertedResidual(nn.Module):
    def __init__(self, in_channel, out_channel, stride, expand_ratio):      # expand_ratio扩展因子，表格中的t
        super(InvertedResidual, self).__init__()
        hidden_channel = in_channel * expand_ratio          # 第一层卷积核的个数=tk，即输入*扩展因子
        self.use_shortcut = stride == 1 and in_channel == out_channel   
            # * use_shortcut 是否使用捷径分支（bool变量）
            # * 要求stride=1（不改变h和w），且输入和输出的shape一样（即保证channel一样即可），这样才能相加

        layers = []
        if expand_ratio != 1:
            # 1x1 pointwise conv
            # 如果扩展因子不为1，那么就有最开始的 1*1 的卷积层
            # * 如果扩展因子为1，即第一个bottleneck出现的情况，那么就相当于没有使用1*1的卷积升高维度；所以实现的时候直接使用了DW卷积，没有最开始的1*1卷积这个步骤
            layers.append(ConvBNReLU(in_channel, hidden_channel, kernel_size=1))
            # 输入是k（即in_channel），输出是t*k （即hidden_channel = in_channel * expand_ratio）；1*1的卷积核 （kernel_size=1）
        layers.extend([             # ^ .extend 列表后添加一个列表，可以批量加入很多层结构
            # 3x3 depthwise conv
            # * DW卷积：输入和输出一样（都是hidden_channel），groups=输入的channel，所以是DW卷积
            ConvBNReLU(hidden_channel, hidden_channel, stride=stride, groups=hidden_channel),
            # 1x1 pointwise conv(linear)
            # * 1*1的普通卷积，降维，使用的是线性激活函数（所以不用ConvBNReLU，而使用nn.Conv2d），线性激活，即y=x，所以这里不添加任何激活函数。
            nn.Conv2d(hidden_channel, out_channel, kernel_size=1, bias=False),   
            # ! 由于有BN层，所以bias=False
            nn.BatchNorm2d(out_channel),        # BN层的输入就等于上一层的输出
        ])

        self.conv = nn.Sequential(*layers)      # ! 这样将列表变为网络

    def forward(self, x):
        if self.use_shortcut:
            return x + self.conv(x)  # 如果有捷径分支，就两者相加 （此时二者维度相同）
        else:
            return self.conv(x)


# ! 下面是 MobileNetV2 网络结构
class MobileNetV2(nn.Module):
    def __init__(self, num_classes=1000, alpha=1.0, round_nearest=8):   # 超参数alpha，控制卷积核个数的倍率
        super(MobileNetV2, self).__init__()
        block = InvertedResidual
        input_channel = _make_divisible(32 * alpha, round_nearest)  
                # *  _make_divisible：将卷积核个数调整为round_nearest的整数倍；这样可以更好地调用设备； 自己进入看看
        last_channel = _make_divisible(1280 * alpha, round_nearest)

        inverted_residual_setting = [    # 该列表对应的就是表格中的每一行的参数：t、c、n、s
            # t, c, n, s   t扩展因子，c输出矩阵的深度，n重复次数、s步距
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        features = []
        # conv1 layer
        features.append(ConvBNReLU(3, input_channel, stride=2))     # input_channel是第一层卷积层的卷积核个数，即当前层的输出维度
        # building inverted residual residual blockes
        for t, c, n, s in inverted_residual_setting:
            output_channel = _make_divisible(c * alpha, round_nearest)  # 调整输出维度，即调整每一层的卷积核个数
            for i in range(n):              # 循环n次；n重复次数
                stride = s if i == 0 else 1     # * 表中的s仅仅表示第一层的步距，所以这样赋值
                features.append(block(input_channel, output_channel, stride, expand_ratio=t))   
                                            # features列表中添加一系列倒残差结构：输入、输出（卷积核个数）、步长、扩展因子t
                input_channel = output_channel      # 维护输入的变量，下一层的输入等于上一层的输出

        # building last several layers
        features.append(ConvBNReLU(input_channel, last_channel, 1))     # 最后的1*1卷积，1280那个
        # * 以上就是所有的特征提取网络部分

        # combine feature layers
        self.features = nn.Sequential(*features)            # 使用Sequential打包成一个网络

        # * building classifier 分类器部分，最后两层
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))         # AdaptiveAvgPool2d自适应的平均下采样操作，输出是1*1
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(last_channel, num_classes)            
        )

        # & weight initialization  初始化权重的流程
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')       # 如果是卷积层，就初始化权重
                if m.bias is not None:
                    nn.init.zeros_(m.bias)                              # 若存在偏置，那么偏置初始化为0
            elif isinstance(m, nn.BatchNorm2d):                         # 如果是BN层，方差设为1，均值设为0
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):                              # 如果是全连接层，权重正态初始化（均值0，方差0.01），偏置设为0
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    # ! 正向传播过程
    def forward(self, x):
        x = self.features(x)            # 特征提取部分
        x = self.avgpool(x)             # 下采样
        x = torch.flatten(x, 1)         # 展平
        x = self.classifier(x)          # 分类器
        return x
