import os
import json

import torch
import torch.nn as nn
from torchvision import transforms, datasets, utils
import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim
from tqdm import tqdm

from model import AlexNet


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")             # torch.device规定训练中所使用的设备
    print("using {} device.".format(device))

    data_transform = {                                                                  # data_transform数据预处理    
        "train": transforms.Compose([transforms.RandomResizedCrop(224),                             # 随机裁剪为224*224
                                     transforms.RandomHorizontalFlip(),                             # 水平方向随机翻转
                                     transforms.ToTensor(),                                         # 转化为tensor     
                                     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),      # 标准化处理
        "val": transforms.Compose([transforms.Resize((224, 224)),               # * cannot 224, must (224, 224)
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])}
    print(os.getcwd())    
    # data_root = os.path.abspath(os.path.join(os.getcwd(), "../.."))  # get data root path           
                                    # 先获取数据集所在的根目录os.getcwd() 
                                    # ^ os.getcwd() 返回当前进程的工作目录，并非当前文件所在的目录
                                    # "../.."表示的是上两层目录，这个要看具体的情况，这是一个相对路径的写法
                                    # ^ os.path.join 路径拼接，拼接后得到的就是当前目录的上两级目录
                                    # ^ os.path.abspath() 获取指定文件或目录的绝对路径（完整路径）

    data_root = os.path.abspath(os.getcwd())
   
    image_path = os.path.join(data_root, "data_set", "flower_data")  # flower data set path
                                        # 等价于 image_path = data_root + "data_set/flower_data"
    # assert os.path.exists(image_path), "{} path does not exist.".format(image_path)

    train_dataset = datasets.ImageFolder(root=os.path.join(image_path, "train"),        # 下载数据集 ，"train"表示是训练集数据   
                                         transform=data_transform["train"])             # 使用"train"的预处理方式
    train_num = len(train_dataset)                                                      # 查看训练集有多少张图片

    # {'daisy':0, 'dandelion':1, 'roses':2, 'sunflower':3, 'tulips':4}
    flower_list = train_dataset.class_to_idx                                            # * .class_to_idx 得到分类名称对应的索引
    cla_dict = dict((val, key) for key, val in flower_list.items())                     # * 将刚刚字典的键值对 变为 值键对
    # write dict into json file
    json_str = json.dumps(cla_dict, indent=4)                                           # 将刚刚的字典变为json形式
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    batch_size = 32
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))

    train_loader = torch.utils.data.DataLoader(train_dataset,                           # 加载数据集
                                               batch_size=batch_size, shuffle=True,     # 通过batchsize和随机参数从样本中获取一批批数据
                                               num_workers=nw)                          # wins下num_workers一般设置为0，linux下num_workers设置可以分布式计算

    validate_dataset = datasets.ImageFolder(root=os.path.join(image_path, "val"),   # root=os.path.join(image_path, "val")等价于 root=image_path+"val"
                                            transform=data_transform["val"])
    val_num = len(validate_dataset)
    validate_loader = torch.utils.data.DataLoader(validate_dataset,
                                                  batch_size=batch_size , shuffle=False,      #   batch_size=4, shuffle=True,    
                                                  num_workers=nw)

    print("using {} images for training, {} images for validation.".format(train_num,
                                                                           val_num))

    # 下面是查看数据集的demo
    # 注意，第60行的batch_size=4, shuffle=True再查看：

    # test_data_iter = iter(validate_loader)
    # test_image, test_label = test_data_iter.next()
    
    # def imshow(img):
    #     img = img / 2 + 0.5  # unnormalize
    #     npimg = img.numpy()
    #     plt.imshow(np.transpose(npimg, (1, 2, 0)))
    #     plt.show()
    
    # print(' '.join('%5s' % cla_dict[test_label[j].item()] for j in range(4)))
    # imshow(utils.make_grid(test_image))


    net = AlexNet(num_classes=5, init_weights=True)             # 5个类别的花数据集，初始化权重为True
                                                                # 实例化模型对象 net
                                                                    
    net.to(device)                                              # ^ net.to(device)将网络放入刚刚指定的设备中
    loss_function = nn.CrossEntropyLoss()                       # 定义损失函数，多类别的交叉熵函数
    # pata = list(net.parameters())                             # 调试所用，查看模型的参数
    optimizer = optim.Adam(net.parameters(), lr=0.0002)         # 定义Adam优化器，优化对象是网络中所有的可训练参数net.parameters(),以及学习了lr=0.0002

    epochs = 10
    save_path = './AlexNet.pth'                                 # 保存权重的路径
    best_acc = 0.0                                              # 最佳准确率 best_acc，首先初始化为0，后面再更新
    train_steps = len(train_loader)

    for epoch in range(epochs):                                 # 迭代10次
                                                                # * 因为使用了dropout，只在训练中使用，预测中不使用   
                                                                          
        # train                                                 #  & 训练阶段
        net.train()                                                     # 调用net.train()进入训练阶段，同时使用 dropout 方法  
        running_loss = 0.0                                              # 统计训练中的平均损失
        train_bar = tqdm(train_loader)                                  # 为了统计训练一个epoch所需时间        
        for step, data in enumerate(train_bar):                         # 遍历数据集；数据集分为图像和标签
            images, labels = data
            optimizer.zero_grad()                                       # 梯度清0
            outputs = net(images.to(device))                            # 正向传播，图像放入设备中，然后实例化AlexNet的网络net中
            loss = loss_function(outputs, labels.to(device))            # 计算损失，计算预测值与真实值的损失，这里label也要放入设备中
            loss.backward()                                             # 反向传播到每一个节点
            optimizer.step()                                            # 更新每一个节点的参数    

            # print statistics
            running_loss += loss.item()                                 # 累加loss值   

            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1,
                                                                     epochs,
                                                                     loss)        # 为了或者训练进度

        # validate                                                  # & 测试阶段
        net.eval()                                                      # 调用net.eval() 进入测试阶段，同时关闭 dropout 方法
        acc = 0.0  # accumulate accurate number / epoch
        with torch.no_grad():                                           # * with torch.no_grad() 禁止参数跟踪：验证中不计算损失梯度            
            val_bar = tqdm(validate_loader)
            for val_data in val_bar:
                val_images, val_labels = val_data                               # 数据划分为图片和对应的标签
                outputs = net(val_images.to(device))                            # 放入网络net中得到输出，输出的维度是 [batch， 10]    
                predict_y = torch.max(outputs, dim=1)[1]                        # 求出输出的第1个维度（dim=1类别维度）max（只关注最大值对应的位置[1]，不关心数值  ），得到预测值 predict_y 
                acc += torch.eq(predict_y, val_labels.to(device)).sum().item()  # 统计预测正确的个数   # ^ 通过.item()得到相应的数值
                # acc += (predict_y == val_labels.to(device)).sum().item()      # 等价的

        val_accurate = acc / val_num                                            # 累加的准确率除以样本个数，得到平均准确率
        print('[epoch %d] train_loss: %.3f  val_accuracy: %.3f' %
              (epoch + 1, running_loss / train_steps, val_accurate))

        if val_accurate > best_acc:                                             # 如果当前准确率大于历史最优准确率
            best_acc = val_accurate                                             # 更新
            torch.save(net.state_dict(), save_path)     

    print('Finished Training')


if __name__ == '__main__':
    main()
