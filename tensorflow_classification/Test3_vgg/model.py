from tensorflow.keras import layers, models, Model, Sequential

# ! 第一种搭建模型方法：使用keras的function ADI 搭建的网络
# ! 分类网络结构
def VGG(feature, im_height=224, im_width=224, num_classes=1000):
    # tensorflow中的tensor通道排序是NHWC
    input_image = layers.Input(shape=(im_height, im_width, 3), dtype="float32")  # Input()` is used to instantiate a Keras tensor.
    x = feature(input_image)                                    # * 输入传入提取特征网络中，得到特征图
    x = layers.Flatten()(x)                                     # 特征进行展平处理
    x = layers.Dropout(rate=0.5)(x)                             # dropout
    x = layers.Dense(2048, activation='relu')(x)                # 原论文中用的是4096个神经元
    x = layers.Dropout(rate=0.5)(x)
    x = layers.Dense(2048, activation='relu')(x)
    x = layers.Dense(num_classes)(x)
    output = layers.Softmax()(x)                                # softmax处理得到类别的概率分布
    model = models.Model(inputs=input_image, outputs=output)    # * models.Model 得到模型，传入输入和输出（概率分布）
    return model

# ! 通过参数列表cfg生成网络结构
# ! 提取特征网络结构
def features(cfg):
    feature_layers = []               # 存储 层结构
    for v in cfg:
        if v == "M":
            feature_layers.append(layers.MaxPool2D(pool_size=2, strides=2))
        else:
            conv2d = layers.Conv2D(v, kernel_size=3, padding="SAME", activation="relu")    # v卷积核个数
            feature_layers.append(conv2d)
    return Sequential(feature_layers, name="feature")        
        # 使用Sequential类，并输入列表就可以得到网络结构


# ! 配置列表字典
cfgs = {
    'vgg11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'vgg19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


# 实例化模型
def vgg(model_name="vgg16", im_height=224, im_width=224, num_classes=1000):
    assert model_name in cfgs.keys(), "not support model {}".format(model_name)
    cfg = cfgs[model_name]
    model = VGG(features(cfg), im_height=im_height, im_width=im_width, num_classes=num_classes)
    return model


# 设置断点并实例化VGG网络
model = vgg(model_name = 'vgg11')