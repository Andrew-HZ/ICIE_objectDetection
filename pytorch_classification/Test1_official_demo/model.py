import torch.nn as nn
import torch.nn.functional as F


class LeNet(nn.Module):                             # ! 定义一个类，继承于nn.Module；并实现两个方法：初始化、前向传播
    def __init__(self):                             # ^ 初始化方法
        super(LeNet, self).__init__()               # 多继承都要用super函数
        self.conv1 = nn.Conv2d(3, 16, 5)            
            # ~ 使用nn.Conv2d定义卷积层: 传入的参数分别为 in_channel，out_channel（等于卷积核的个数），kernel_size， stride=1， padding =0， 
        self.pool1 = nn.MaxPool2d(2, 2)             # ~ 使用nn.MaxPool2d：kernel_size， stride
        self.conv2 = nn.Conv2d(16, 32, 5)           # 输入的深度和上一层的输出一样16，使用32个5*5的卷积核
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fc1   = nn.Linear(32*5*5, 120)         # ~ 使用nn.Linear：in_channel，out_channel
        self.fc2   = nn.Linear(120, 84)
        self.fc3   = nn.Linear(84, 10)              # 最后的输出要根据类别进行修改

    def forward(self, x):                           # ^ 定义正向传播的过程
        x = F.relu(self.conv1(x))    # input(3, 32, 32) output(16, 28, 28)，有16个卷积核所以输出的channel是16； 计算(32-5+2*0)/1+1=28
        x = self.pool1(x)            # output(16, 14, 14)，channel深度不变，w和h各缩小一半
        x = F.relu(self.conv2(x))    # output(32, 10, 10)； (14-5+2*0)/1+1=10
        x = self.pool2(x)            # output(32, 5, 5)
        x = x.view(-1, 32*5*5)       # output(32*5*5)        # ! .view展开
        x = F.relu(self.fc1(x))      # output(120)
        x = F.relu(self.fc2(x))      # output(84)
        x = self.fc3(x)              # output(10)           # 内部已经实现了softmax（softmax转变为概率）
        return x



import torch

if __name__ == '__main__':
    input1 = torch.rand([32,3,32,32])           # ! tensor的维度都是 batch， channel， H， W
    model = LeNet()                             # ^ 开始实例化模型
    print(model)
    output = model(input1)            
