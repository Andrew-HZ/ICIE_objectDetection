import os
import json

import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

from model_v2 import MobileNetV2


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    data_transform = transforms.Compose(                    # 预处理部分
        [transforms.Resize(256),
         transforms.CenterCrop(224),
         transforms.ToTensor(),
         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    # load image
    img_path = "../tulip.jpg"
    img_path = ".\\pytorch_classification\\Test6_mobilenet\\tulip.jpg"

    assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
    img = Image.open(img_path)
    plt.imshow(img)
    # [N, C, H, W]
    img = data_transform(img)
    # expand batch dimension
    img = torch.unsqueeze(img, dim=0)       # * 添加一个batch维度

    # read class_indict
    json_path = './class_indices.json'
    assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)

    json_file = open(json_path, "r")
    class_indict = json.load(json_file)

    # * create model 实例化模型
    model = MobileNetV2(num_classes=5).to(device)


    # * load model weights 载入模型权重
    # model_weight_path = "./MobileNetV2.pth"
    model_weight_path = ".\\pytorch_classification\\Test6_mobilenet\\mobilenet_v2.pth"
    
    model.load_state_dict(torch.load(model_weight_path, map_location=device))
    model.eval()
    with torch.no_grad():           # * 禁止更新梯度
        # predict class
        output = torch.squeeze(model(img.to(device))).cpu()  # * squeeze压缩batch维度
        predict = torch.softmax(output, dim=0)               # * softmax将输出转化为概率分布
        predict_cla = torch.argmax(predict).numpy()          # 获得最大的预测值所对应的索引   

    print_res = "class: {}   prob: {:.3}".format(class_indict[str(predict_cla)],
                                                 predict[predict_cla].numpy())
    plt.title(print_res)
    print(print_res)
    plt.show()


if __name__ == '__main__':
    main()
