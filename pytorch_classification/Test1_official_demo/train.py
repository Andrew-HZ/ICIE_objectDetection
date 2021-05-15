import torch
import torchvision
import torch.nn as nn
from model import LeNet
import torch.optim as optim
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt


# ! 先将 20~69行（main中除了 train_set的都注释掉，download=True 运行），就可以先下载数据集了
def main():
    transform = transforms.Compose(                                     #  ^ transforms.Compose将使用的预处理方法打包成一个整体
        [transforms.ToTensor(),                                         
                # ToTensor自己查看定义：将 H*W*C的0~255的图像变为 C*H*W的0~1的张量
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])       
                # 标准化，  Given mean: ``(mean[1],...,mean[n])`` and std: ``(std[1],..,std[n])`` for ``n``

    # 50000张训练图片
    # 第一次使用时要将download设置为True才会自动去下载数据集
    # torchvision.datasets里面有很多数据集，这里使用的是CIFAR10
    train_set = torchvision.datasets.CIFAR10(root='./data', train=True,             # root指定位置；train=True表示导入训练集
                                             download=False, transform=transform)   # 使用的预处理方法transform
    
    # 将训练集导入，并分为一个个批次
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=36,
                                               shuffle=True, num_workers=0)        
                # shuffle=True是打乱，随机提取到batch； # ^ num_workers载入的线程数，win下只能为0

    # 10000张验证图片
    # 第一次使用时要将download设置为True才会自动去下载数据集
    val_set = torchvision.datasets.CIFAR10(root='./data', train=False,              # root指定位置；train=False表示导入测试集
                                           download=False, transform=transform)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=5000,
                                             shuffle=False, num_workers=0)

    val_data_iter = iter(val_loader)                # ^ iter 变成一个可迭代对象；然后通过.next()就可以获取到一批数据
    val_image, val_label = val_data_iter.next()     # 得到了测试图像及对应的标签值
    
    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')   # 元祖类型


    #  ! # 如果想查看图片，使用下面的代码;
    #     # 参考官方文档 https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html#sphx-glr-beginner-blitz-cifar10-tutorial-py

    # def imshow(img):
    #     img = img / 2 + 0.5     # unnormalize，对图像反标准化处理； transforms.Normalize是对图像均值-0.5，方差是0.5；所以这里反向操作
    #     npimg = img.numpy()     # 将图像从tensor变为numpy格式
    #     plt.imshow(np.transpose(npimg, (1, 2, 0)))      # 维度还原，tensor的C*H*W变为numpy的H*W*C
    #     plt.show()
    

    # # print labels
    # # print(' '.join('%5s' % classes[labels[j]] for j in range(batch_size)))
    # print(' '.join('%5s' % classes[val_label[j]] for j in range(4)))  # 不要太大了，就显示4张图片

    # # show images
    # # imshow(torchvision.utils.make_grid(images))
    # imshow(torchvision.utils.make_grid(val_image)) 


    net = LeNet()                                       # 实例化模型
    loss_function = nn.CrossEntropyLoss()               # 定义损失函数，CrossEntropyLoss里面其实就有了softmax函数
        # This criterion combines :class:`~torch.nn.LogSoftmax` and :class:`~torch.nn.NLLLoss` in one single class
    optimizer = optim.Adam(net.parameters(), lr=0.001)  # 定义优化器：传入需要训练的网络参数net.parameters()

    for epoch in range(5):                              # loop over the dataset multiple times; epoch表示迭代多少轮

        running_loss = 0.0                              # running_loss累计损失
        for step, data in enumerate(train_loader, start=0):         # 遍历训练集样本
            # enumerate函数，不仅返回每一个批次的data，还会返回对应的索引；start=0，索引从0开始
            # get the inputs; data is a list of [inputs, labels]

            inputs, labels = data                       # 分离成图像和标签

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)                       # 正向传播，得到输出    
            loss = loss_function(outputs, labels)       # 通过真实值和输出值得到loss
            loss.backward()                             # 通过loss进行反向传播
            optimizer.step()                            # 使用optimizer.step()更新参数

            # print statistics 打印过程
            running_loss += loss.item()                 # 每次计算好loss，都累加如running_loss
            if step % 500 == 499:                       # print every 500 mini-batches；每500步打印一次信息
                with torch.no_grad():                   # 接下来不要计算每个节点误差损失梯度（测试集）
                        # ~ 这样可以节省算力和内存；如果不用torch.no_grad这个函数，在测试阶段内存很可能崩溃
                    outputs = net(val_image)            # 输出的维度为[batch, 10]，第0个维度表示batch，第1个维度是类别
                    predict_y = torch.max(outputs, dim=1)[1]  # 在第1个维度上找到输出最大的index
                                                        # ^ max返回两个值，第0个是数值，第1个是位置；所以有 [1]   
                    accuracy = torch.eq(predict_y, val_label).sum().item() / val_label.size(0)
                                                        # 比较预测的标签类别和真实的标签类别，并统计求和
                                                        # ^ 都是在tensor中计算的，所以通过.item()得到相应的数值
                                                        # 最后除以测试样本的数目得到准确率

                    print('[%d, %5d] train_loss: %.3f  test_accuracy: %.3f' %
                          (epoch + 1, step + 1, running_loss / 500, accuracy)) 
                                # 训练的第epoch轮，该轮的step步，平均训练误差（因为每500步打印一次），准确率
                    running_loss = 0.0                  # running_loss清零，进行下个500steps的测试

    print('Finished Training')                          # 全部训练完后，打印该信息

    save_path = './Lenet.pth'
    torch.save(net.state_dict(), save_path)             # 保存训练的模型，torch.save保存所有的参数


if __name__ == '__main__':
    main()
