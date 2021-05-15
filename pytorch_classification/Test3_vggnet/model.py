import torch.nn as nn
import torch

# official pretrain weights
model_urls = {
    'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
    'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth'
}


class VGG(nn.Module):
    def __init__(self, features, num_classes=1000, init_weights=False):   # features提取的特征，是一个函数参数
        super(VGG, self).__init__()
        self.features = features
        self.classifier = nn.Sequential(                      # 分类网络，3个全连接层
            nn.Linear(512*7*7, 4096),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, num_classes)
        )
        if init_weights:
            self._initialize_weights()                        # init_weights是否权重初始化 

    def forward(self, x):
        # N x 3 x 224 x 224
        x = self.features(x)                                  # 说明了features是一个函数参数；传入数据x得到特征
        # N x 512 x 7 x 7
        x = torch.flatten(x, start_dim=1)                     # 展平操作，start_dim表示从哪个维度开始展平；维度依次是B*C*H*W 
        # N x 512*7*7       
        x = self.classifier(x)                                # 特征矩阵再传入分类网络中  
        return x

    def _initialize_weights(self):
        for m in self.modules():                              # 遍历每一层      
            if isinstance(m, nn.Conv2d):
                # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.xavier_uniform_(m.weight)             # 卷积层用xavier_uniform_初始化  
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)              # 如有偏置，设为0  
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                # nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


# & 提取特征网络
def make_features(cfg: list):                                               # 传入配置变量，list类型
                    # ^ 函数参数中的冒号是参数的类型建议符，告诉程序员希望传入的实参的类型
                    # ^ 函数后面跟着的箭头是函数返回值的类型建议符，用来说明该函数返回的值是什么类型
    layers = []                                                             # layers 空列表存放每一层的结构
    in_channels = 3
    for v in cfg:                                                           # for loop；遍历配置列表
        if v == "M":                                                        # 如果是'M'，该层就是最大池化层
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)    # 输入的是in_channels，输出的是v（卷积核的个数）
            layers += [conv2d, nn.ReLU(True)]                               # 将conv2d, nn.ReLU加入layers中
            in_channels = v                                                 # 这样输出就变为v了，同时是下一层的输入
    return nn.Sequential(*layers)                                           # nn.Sequential 非关键字传递   

"""  # Example of using Sequential  补充一下，Sequential有两种传参形式； 看定义
        model = nn.Sequential(
                  nn.Conv2d(1,20,5),
                  nn.ReLU(),
                  nn.Conv2d(20,64,5),
                  nn.ReLU()
                )

        # Example of using Sequential with OrderedDict
        model = nn.Sequential(OrderedDict([
                  ('conv1', nn.Conv2d(1,20,5)),
                  ('relu1', nn.ReLU()),
                  ('conv2', nn.Conv2d(20,64,5)),
                  ('relu2', nn.ReLU())
                ])) """

# & 定义一个字典，key是模型的配置文件；value是列表，里面的数字对应的是卷积核的个数，'M'表示池化层
cfgs = {
    'vgg11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'vgg19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}



def vgg(model_name="vgg16", **kwargs):
    assert model_name in cfgs, "Warning: model number {} not in cfgs dict!".format(model_name)
    cfg = cfgs[model_name]                              # 传入key，得到配置列表

    model = VGG(make_features(cfg), **kwargs)           # 实例化，**kwargs表示可变长度的字典变量
    return model


# vgg_model = vgg(model_name='vgg13')