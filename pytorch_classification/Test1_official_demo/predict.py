import torch
import torchvision.transforms as transforms
from PIL import Image                       

from model import LeNet


def main():
    transform = transforms.Compose(
        [transforms.Resize((32, 32)),                                # 因为下载的图片大小不一，所以先resize      
         transforms.ToTensor(),                                      # ToTensor() 将 H*W*C的numpy 变为 C*H*W的tensor
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])    # 标准化处理

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    net = LeNet()                                           # 实例化对象
    net.load_state_dict(torch.load('Lenet.pth'))            # 通过load_state_dict载入保存的权重文件

    #im = Image.open('1.jpg')                               # PIL库的Image 打开图像，H*W*C
    im = Image.open('./pytorch_classification/Test1_official_demo/1.jpg')                                            
    im = transform(im)  # [C, H, W]                         # 如果要在网络中传播，必须变为tensor格式，所以transform变为C*H*W
    im = torch.unsqueeze(im, dim=0)  # [N, C, H, W]         # ^ torch.unsqueeze(~~~ , dim=0)在第0维增加新的维度；变为N*C*H*W

    with torch.no_grad():                                   # ^ 测试阶段，不需要求损失梯度，使用torch.no_grad()
        outputs = net(im)                                   # 图像传入网络中得到输出，输出的维度为[batch, 10]，只关注dim=1
        predict = torch.max(outputs, dim=1)[1].data.numpy() # 找到第1个维度中最大的数，但是只关注其位置（即[1]，index），转化为numpy  
        predict1 = torch.softmax(outputs, dim=1)            # 得到了概率分布
    print(classes[int(predict)])                            # 将index传入到classes就得到了类别     
    print(predict1)                                         # 打印概率分布

if __name__ == '__main__':
    main()
