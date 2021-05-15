import torch.nn as nn
import torch


class AlexNet(nn.Module):
    def __init__(self, num_classes=1000, init_weights=False):
        super(AlexNet, self).__init__()

        self.features = nn.Sequential(                              
                    # ~ 使用nn.Sequential将一系列的层结构进行打包；起名为features（提取特征网络）
                    #  对比之前的，写的是  self.conv1 = nn.Conv2d(3, 48, kernel_size=11, stride=4, padding=2)；
                    #  有了nn.Sequential，现在能够直接写  nn.Conv2d(3, 48, kernel_size=11, stride=4, padding=2)； 
                    # ^ 不用再写 self.名称 = nn.Conv2d(); 可以直接写  nn.Conv2d()了； 
                    # ^ 层数比较多的时候使用nn.Sequential可以大大精简代码；  
                                         
            nn.Conv2d(3, 48, kernel_size=11, stride=4, padding=2),  
                    # input[3, 224, 224]  output[48, 55, 55]； 这里卷积核个数取96的一半48；
                    # 卷积的padding可以传入两个形式的变量，要么是int（上下左右都补充int行0），要么就是tuple类型
                    # * 比如padding传入的是 tuple:(1,2)，1代表上下方各补一行0,2代表左右两侧各补两列0；那么就使用官方的nn.ZeroPad2d() 自己看代码
                    # * 这里为了简便，直接写为 padding=2；这样计算的结果是55.25为小数，当不为整数的时候，会自动的舍弃最后一行和最后一列，那么也就是右侧和底部都只补充了一层0，就和之前的操作一样了。 （参考CSDN "pytorch中的卷积操作详解"）

            nn.ReLU(inplace=True),                                  # inplace参数可以增加计算量，减少内存耗用
            nn.MaxPool2d(kernel_size=3, stride=2),                  # output[48, 27, 27]
            nn.Conv2d(48, 128, kernel_size=5, padding=2),           # output[128, 27, 27] 下面的输出都只取了一半，因为作者当成就是这么做的
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),                  # output[128, 13, 13]
            nn.Conv2d(128, 192, kernel_size=3, padding=1),          # output[192, 13, 13]
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 192, kernel_size=3, padding=1),          # output[192, 13, 13]
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 128, kernel_size=3, padding=1),          # output[128, 13, 13]
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),                  # output[128, 6, 6]
        )

        self.classifier = nn.Sequential(                            # 分类器
            nn.Dropout(p=0.5),                                      # ^ 使用nn.Dropout使部分神经元失活，通常用于全连接层之间；p是随机失活的比例
                                                                    # 这里是在展平操作之后，与后面的全连接层之间加入dropout函数
            nn.Linear(128 * 6 * 6, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),                                      # 这两个fc层之间也使用了nn.Dropout
            nn.Linear(2048, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, num_classes),                           # 输出的是类别的数量
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x):                                       # ~ 正向传播函数       
        x = self.features(x)                                        # 输入x传入特征提取网络，得到特征图x
        x = torch.flatten(x, start_dim=1)                           # 展平torch.flatten，从第1维开始展平
                                                                    # 因为tensor的维度是B*C*H*W，batch维度不用管，直接从channel维度开始展平   
        x = self.classifier(x)                                      # 然后传入分类器中
        return x


    def _initialize_weights(self):                               # ~ 初始化权重的函数  （要了解，其实是自动初始化的）
        for m in self.modules():                                    # ^ modules()返回一个迭代器，遍历网络中的所有的模块

            if isinstance(m, nn.Conv2d):                         # * 如果是nn.Conv2d：
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu') 
                                                                    # * 使用kaiming_normal_方法（何凯明提出的）对权重初始化
                if m.bias is not None:                              # 如果偏置不为空，用0初始化
                    nn.init.constant_(m.bias, 0)                    

            elif isinstance(m, nn.Linear):                       # * 如果是nn.Linear：
                nn.init.normal_(m.weight, 0, 0.01)                  # * 使用正态分布对权重初始化
                nn.init.constant_(m.bias, 0)                        # 偏置全初始化为0   
