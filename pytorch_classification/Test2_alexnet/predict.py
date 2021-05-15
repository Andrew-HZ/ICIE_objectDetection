import os
import json

import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

from model import AlexNet


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    data_transform = transforms.Compose(                                                # 首先对图片预处理： resize+toTensor+normalize        
        [transforms.Resize((224, 224)),
         transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    # load image
    img_path = ".\\pytorch_classification\\Test2_alexnet.\\dandelion.jpg"                                                           # 载入图片    
    assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
    img = Image.open(img_path)                  

    plt.imshow(img)
    # [N, C, H, W]
    img = data_transform(img)
    # expand batch dimension
    img = torch.unsqueeze(img, dim=0)                                           # ^ .unsqueeze 扩充一个维度（batch维度）
                                                                                # ^ .squeeze() 对数据的维度进行压缩，去掉维数为1的的维度

    # read class_indict
    json_path = './class_indices.json'                                                  # 读取json文件，即索引对应的类别名称
    assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)

    json_file = open(json_path, "r")
    class_indict = json.load(json_file)

    # create model
    model = AlexNet(num_classes=5).to(device)                                           # 初始化网络并放入设备中

    # load model weights
    weights_path = "./AlexNet.pth"
    assert os.path.exists(weights_path), "file: '{}' dose not exist.".format(weights_path)
    model.load_state_dict(torch.load(weights_path))                                     # 载入权重

    model.eval()                                                                        # 进入测试模式（包含dropout操作）
    with torch.no_grad():                                       # with torch.no_grad() 禁止参数跟踪：验证中不计算损失梯度                   
        # predict class
        output = torch.squeeze(model(img.to(device))).cpu()     # torch.squeeze压缩了维数为1的的维度（batch维度）   
        predict = torch.softmax(output, dim=0)                  # 通过softmax变成概率分布
        predict_cla = torch.argmax(predict).numpy()             # 获得概率最大的那个索引值，并将其转化为numpy

    print_res = "class: {}   prob: {:.3}".format(class_indict[str(predict_cla)],            # 打印类别名称和预测概率
                                                 predict[predict_cla].numpy())
    plt.title(print_res)
    print(print_res)
    plt.show()


if __name__ == '__main__':
    main()
