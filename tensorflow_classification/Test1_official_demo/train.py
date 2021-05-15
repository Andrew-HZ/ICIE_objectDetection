from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
from model import MyModel


def main():
    mnist = tf.keras.datasets.mnist

    # download and load data
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    # Add a channels dimension
    x_train = x_train[..., tf.newaxis]                          # 现在增加了一个维度；可以debug，array变为了28*28*1的维度
    x_test = x_test[..., tf.newaxis]                              # 添加了灰度这个维度  

    # create data generator                                     # 创建数据生成器
    train_ds = tf.data.Dataset.from_tensor_slices(
        (x_train, y_train)).shuffle(10000).batch(32)            # 将图像和标签合并为元组，取10000张图片打乱，每个批次提取32张
                                                                # shuffle的参数越接近数据集的大小，越接近随机采用；但是受内存影响，一般就取很大的数    
    test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)

    # create model
    model = MyModel()                                                   # 定义模型

    # define loss
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy()       # 定义loss；打开看一下定义，这个是直接比较标签类别编码，而不是one-hot编码
                                                                        # SparseCategoricalCrossentropy稀疏的多类别交叉熵损失    
    # define optimizer
    optimizer = tf.keras.optimizers.Adam()                              # 定义优化器

    # define train_loss and train_accuracy                              # ^ 使用相应的函数计算损失和准确率    
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

    # define train_loss and train_accuracy
    test_loss = tf.keras.metrics.Mean(name='test_loss')
    test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

    # define train function including calculating loss, applying gradient and calculating accuracy
    # ~ tf.function装饰器可以构建高效的python代码，将代码转化为图结构，在CPU\GPU上快速运行
    # * 加了这个装饰器，就不能在这里面设置断点进行调试了；
    @tf.function
    def train_step(images, labels):                                             # pytorvh可以自动跟踪每一个可循了参数的误差梯度                    
        with tf.GradientTape() as tape:                                         # tensorflow中不会自动跟踪，使用 with tf.GradientTape()    
            predictions = model(images)                                         # 输入图片传入模型中得到输出
            loss = loss_object(labels, predictions)                             # 计算损失
        gradients = tape.gradient(loss, model.trainable_variables)              # 通过tf.GradientTape将损失传到模型的变量中
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))    # 用误差梯度更新优化值；注意：这里用梯度和参数值使用zip打包成元祖输入

        train_loss(loss)                                                        # loss传入累加器中，计算历史的平均loss
        train_accuracy(labels, predictions)                                     # accuracy传入累加器中，计算历史的平均accuracy

    # define test function including calculating loss and calculating accuracy
    @tf.function
    def test_step(images, labels):                                              # 测试中不用跟踪梯度，所以不用 with tf.GradientTape()    
        predictions = model(images)
        t_loss = loss_object(labels, predictions)

        test_loss(t_loss)
        test_accuracy(labels, predictions)

    EPOCHS = 5

    for epoch in range(EPOCHS):
        train_loss.reset_states()        # clear history info   # ^ 每次新的一轮，这些值全部清0
        train_accuracy.reset_states()    # clear history info
        test_loss.reset_states()         # clear history info
        test_accuracy.reset_states()     # clear history info

        for images, labels in train_ds:     # 遍历训练迭代器
            train_step(images, labels)

        for test_images, test_labels in test_ds:
            test_step(test_images, test_labels)

        template = 'Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}'     # 每训练一个epoch，都打印相关的信息
        print(template.format(epoch + 1,
                              train_loss.result(),                      # .result()可以获取该函数的值
                              train_accuracy.result() * 100,
                              test_loss.result(),
                              test_accuracy.result() * 100))


if __name__ == '__main__':
    main()






# # 可以注释，然后先载入数据集

# from __future__ import absolute_import, division, print_function, unicode_literals

# import tensorflow as tf
# from model import MyModel


# def main():
#     mnist = tf.keras.datasets.mnist

#     # download and load data
#     (x_train, y_train), (x_test, y_test) = mnist.load_data()
#     x_train, x_test = x_train / 255.0, x_test / 255.0

# if __name__ == '__main__':
#     main()




# # 可以注释，然后先载入数据集

# from __future__ import absolute_import, division, print_function, unicode_literals

# import tensorflow as tf
# from model import MyModel
# import numpy as np
# import matplotlib.pyplot as plt


# def main():
#     mnist = tf.keras.datasets.mnist

#     # download and load data
#     (x_train, y_train), (x_test, y_test) = mnist.load_data()
#     x_train, x_test = x_train / 255.0, x_test / 255.0

#     imgs =  x_test[:3]
#     labs = y_test
#     print(labs)
#     plot_imgs = np.hstack(imgs)   # hstack水平方向拼接，h是水平的意思
#     plt.imshow(plot_imgs, cmap='gray')  # 灰度图片
#     plt.show()

# if __name__ == '__main__':
#     main()
