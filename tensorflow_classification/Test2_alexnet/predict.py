import os
import json

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

from model import AlexNet_v1, AlexNet_v2


def main():
    im_height = 224
    im_width = 224

    # load image
    # img_path = "../tulip.jpg"
    img_path = ".\\tensorflow_classification\\Test2_alexnet\\11.jpg"
    assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
    img = Image.open(img_path)                                  # 打开图片

    # resize image to 224x224
    img = img.resize((im_width, im_height))                     # resize处理：224*224
    plt.imshow(img)

    # scaling pixel value to (0-1)
    img = np.array(img) / 255.                                  # 缩放到0~1之间

    # Add the image to a batch where it's the only member.
    img = (np.expand_dims(img, 0))                              # 增加一个维度，B*H*W*C   # ^ np.expand_dims(img, 0) 最前面扩充一个维度

    # read class_indict
    json_path = './class_indices.json'
    assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)

    json_file = open(json_path, "r")
    class_indict = json.load(json_file)                         # 读取json文件

    # create model
    model = AlexNet_v1(num_classes=5)                           # 实例化模型
    weighs_path = "./save_weights/myAlex.h5"
    assert os.path.exists(img_path), "file: '{}' dose not exist.".format(weighs_path)
    model.load_weights(weighs_path)                             # ^ .load_weights 载入模型

    # prediction
    result = np.squeeze(model.predict(img))                     # model.predict预测图片；并去掉batch维度 np.squeeze，得到概率分布
    predict_class = np.argmax(result)                           # 获取概率最大的值对应的索引           

    print_res = "class: {}   prob: {:.3}".format(class_indict[str(predict_class)],      # * 通过json文件得到所属类别
                                                 result[predict_class])
    plt.title(print_res)
    print(print_res)
    plt.show()


if __name__ == '__main__':
    main()
