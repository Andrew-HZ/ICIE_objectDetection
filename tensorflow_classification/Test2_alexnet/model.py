from tensorflow.keras import layers, models, Model, Sequential                   
    # ^ keras里面的layers, models, Model, Sequential

# ! 第一种搭建模型方法：使用keras的function ADI 搭建的网络
def AlexNet_v1(im_height=224, im_width=224, num_classes=1000):
    # tensorflow中的tensor通道排序是NHWC
    input_image = layers.Input(shape=(im_height, im_width, 3), dtype="float32")  # output(None, 224, 224, 3)
    x = layers.ZeroPadding2D(((1, 2), (1, 2)))(input_image)                      # output(None, 227, 227, 3)
        # * same或者valid padding都没法通过224*224变为55*55，所以先用ZeroPadding2D扩充为227*227，然后valid padding变为55*55
        # * interpreted as ((top_pad, bottom_pad), (left_pad, right_pad))

    x = layers.Conv2D(48, kernel_size=11, strides=4, activation="relu")(x)       # output(None, 55, 55, 48) 默认valid padding
    x = layers.MaxPool2D(pool_size=3, strides=2)(x)                              # output(None, 27, 27, 48)
    x = layers.Conv2D(128, kernel_size=5, padding="same", activation="relu")(x)  # output(None, 27, 27, 128)
    x = layers.MaxPool2D(pool_size=3, strides=2)(x)                              # output(None, 13, 13, 128)
    x = layers.Conv2D(192, kernel_size=3, padding="same", activation="relu")(x)  # output(None, 13, 13, 192)
    x = layers.Conv2D(192, kernel_size=3, padding="same", activation="relu")(x)  # output(None, 13, 13, 192)
    x = layers.Conv2D(128, kernel_size=3, padding="same", activation="relu")(x)  # output(None, 13, 13, 128)
    x = layers.MaxPool2D(pool_size=3, strides=2)(x)                              # output(None, 6, 6, 128)

    x = layers.Flatten()(x)                         # output(None, 6*6*128)      # ^ 全连接之前，要将特征矩阵展成一维向量 .Flatten()
    x = layers.Dropout(0.2)(x)                      # 20% 失活比例                # ^ .Dropout在使用全连接层之前                      
    x = layers.Dense(2048, activation="relu")(x)    # output(None, 2048)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(2048, activation="relu")(x)    # output(None, 2048)
    x = layers.Dense(num_classes)(x)                # output(None, 5)            # ^ 最后一个Dense不要设置激活函数，因为最后有softmax处理
    predict = layers.Softmax()(x)                                                # softmax将输出转化为概率分布             

    model = models.Model(inputs=input_image, outputs=predict)                    # ^ 通过 models.Model 定义网络的 输入 和 输出
    return model


# ! 第二种搭建模型方法：使用子类搭建的网络； 类似于pytorch
class AlexNet_v2(Model):
    def __init__(self, num_classes=1000):
        super(AlexNet_v2, self).__init__()
        self.features = Sequential([
            layers.ZeroPadding2D(((1, 2), (1, 2))),                                 # output(None, 227, 227, 3)
            layers.Conv2D(48, kernel_size=11, strides=4, activation="relu"),        # output(None, 55, 55, 48)
            layers.MaxPool2D(pool_size=3, strides=2),                               # output(None, 27, 27, 48)
            layers.Conv2D(128, kernel_size=5, padding="same", activation="relu"),   # output(None, 27, 27, 128)
            layers.MaxPool2D(pool_size=3, strides=2),                               # output(None, 13, 13, 128)
            layers.Conv2D(192, kernel_size=3, padding="same", activation="relu"),   # output(None, 13, 13, 192)
            layers.Conv2D(192, kernel_size=3, padding="same", activation="relu"),   # output(None, 13, 13, 192)
            layers.Conv2D(128, kernel_size=3, padding="same", activation="relu"),   # output(None, 13, 13, 128)
            layers.MaxPool2D(pool_size=3, strides=2)])                              # output(None, 6, 6, 128)

        self.flatten = layers.Flatten()
        self.classifier = Sequential([
            layers.Dropout(0.2),
            layers.Dense(1024, activation="relu"),                                  # output(None, 2048)
            layers.Dropout(0.2),
            layers.Dense(128, activation="relu"),                                   # output(None, 2048)
            layers.Dense(num_classes),                                                # output(None, 5)
            layers.Softmax()
        ])

    def call(self, inputs, **kwargs):
        x = self.features(inputs)
        x = self.flatten(x)
        x = self.classifier(x)
        return x
