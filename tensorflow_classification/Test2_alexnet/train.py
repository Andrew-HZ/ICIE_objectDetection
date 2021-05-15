from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from model import AlexNet_v1, AlexNet_v2
import tensorflow as tf
import json
import os


def main():
    # * 定义训练集和测试集文件的位置
    # data_root = os.path.abspath(os.path.join(os.getcwd(), "../.."))  # get data root path
    data_root = os.path.abspath(os.getcwd()) # get data root path
    image_path = os.path.join(data_root, "data_set", "flower_data")  # flower data set path
    train_dir = os.path.join(image_path, "train")
    validation_dir = os.path.join(image_path, "val")
    assert os.path.exists(train_dir), "cannot find {}".format(train_dir)
    assert os.path.exists(validation_dir), "cannot find {}".format(validation_dir)

    # create direction for saving weights
    if not os.path.exists("save_weights"):
        os.makedirs("save_weights")

    # * 网络的基本参数：输入图像的H、W；batch_size；epoch
    im_height = 224             
    im_width = 224
    batch_size = 32
    epochs = 10

    # data generator with data augmentation
    # from tensorflow.keras.preprocessing.image import ImageDataGenerator 图像生成器
    # * ImageDataGenerator()有很多图像预处理方法，自己查看
    # 分别定义训练集和验证集的图像生成器
    train_image_generator = ImageDataGenerator(rescale=1. / 255,                # 并从0~255缩放到0~1之间
                                               horizontal_flip=True)            # 随机水平翻转    
    validation_image_generator = ImageDataGenerator(rescale=1. / 255)       

    # * 通过.flow_from_directory读取图片
    train_data_gen = train_image_generator.flow_from_directory(directory=train_dir,
                                                               batch_size=batch_size,
                                                               shuffle=True,
                                                               target_size=(im_height, im_width),
                                                               class_mode='categorical')     # categorical分类特征
    # * .n 获得训练集样本的个数
    total_train = train_data_gen.n                 

    # * get class dict； .class_indices获得类别名称的index
    class_indices = train_data_gen.class_indices

    # transform value and key of dict
    # *	遍历字典，键值对 转化为 值键对
    inverse_dict = dict((val, key) for key, val in class_indices.items())   

    # * write dict into json file  将转化后的字典写入json文件中
    json_str = json.dumps(inverse_dict, indent=4)       
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)


    # 同样的方法，设置验证集生成器类似的信息
    val_data_gen = validation_image_generator.flow_from_directory(directory=validation_dir,
                                                                  batch_size=batch_size,
                                                                  shuffle=False,
                                                                  target_size=(im_height, im_width),
                                                                  class_mode='categorical')
    total_val = val_data_gen.n
    print("using {} images for training, {} images for validation.".format(total_train,
                                                                           total_val))



    # # 下面看一下载入的图像信息   可注释
    # sample_training_images, sample_training_labels = next(train_data_gen)  # label is one-hot coding
    # # * 注意：ImageDataGenerator()生成器会自动地将label转化为 one-hot编码形式
    
    # # This function will plot images in the form of a grid with 1 row
    # # and 5 columns where images are placed in each column.
    # def plotImages(images_arr):
    #     fig, axes = plt.subplots(1, 5, figsize=(20, 20))
    #     axes = axes.flatten()
    #     for img, ax in zip(images_arr, axes):
    #         ax.imshow(img)
    #         ax.axis('off')
    #     plt.tight_layout()
    #     plt.show()
    
    # plotImages(sample_training_images[:5])   # 查看前5张数据


    model = AlexNet_v1(im_height=im_height, im_width=im_width, num_classes=5)


    # model = AlexNet_v2(class_num=5)
    # model.build((batch_size, 224, 224, 3))  # when using subclass model
    # & 如果使用的是子类模型的方法构建网络，在model.summary() 之前必须加上model.build()，build调用后才真正实例化这个模型了

    model.summary()          # ^ model.summary()可以看到模型的参数信息



    # using keras high level api for training
    # ! 使用keras的高层API方法进行训练
    # * 首先对模型进行编译 model.compile()，定义优化器、loss计算、要打印的信息
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
                  loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),  
                        # from_logits=False，因为搭建模型的时候已经用到了softmax处理
                        # CategoricalCrossentropy expect labels to be provided in a `one_hot` representation；进入函数查看
                        # If you want to provide labels as integers, please use `SparseCategoricalCrossentropy` loss
                  metrics=["accuracy"])             # ^ metrics 表示需要监控的指标

    # 定义回调函数列表
    # * .callbacks.ModelCheckpoint 定义保存模型的一些参数
    callbacks = [tf.keras.callbacks.ModelCheckpoint(filepath='./save_weights/myAlex.h5',
                # &  保存模型的格式有两种： keras的官方 .h5 模式；tensorflow的ckpt格式； 自己改后缀就可以
                                                    save_best_only=True,        # 是否只保存最佳的参数
                                                    save_weights_only=True,     # 是否只保存权重文件，那么参数会小；但是如果要用，就要自己先创建模型，然后再载入权重；  如果还保存了模型文件，就不用自己创建了，直接调用文件就可以
                                                    monitor='val_loss')]        # monitor 监控验证集的损失，以此判断当前是否为最佳参数

    # * tensorflow2.1 recommend to using fit，推荐使用fit，保存在history中
    # tf2.1之后，model.fit已经兼容了model.fit_generator
    history = model.fit(x=train_data_gen,                               # x是训练集输入，train_data_gen训练集生成器
                        steps_per_epoch=total_train // batch_size,      # steps_per_epoch每一轮迭代次数
                        epochs=epochs,
                        validation_data=val_data_gen,
                        validation_steps=total_val // batch_size,
                        callbacks=callbacks)                            # * callbacks回调函数，即保存模型的规则

    # plot loss and accuracy image
    history_dict = history.history                                      # history.history得到数据字典
    train_loss = history_dict["loss"]
    train_accuracy = history_dict["accuracy"]
    val_loss = history_dict["val_loss"]
    val_accuracy = history_dict["val_accuracy"]

    # figure 1
    plt.figure()
    plt.plot(range(epochs), train_loss, label='train_loss')
    plt.plot(range(epochs), val_loss, label='val_loss')
    plt.legend()
    plt.xlabel('epochs')
    plt.ylabel('loss')

    # figure 2
    plt.figure()
    plt.plot(range(epochs), train_accuracy, label='train_accuracy')
    plt.plot(range(epochs), val_accuracy, label='val_accuracy')
    plt.legend()
    plt.xlabel('epochs')
    plt.ylabel('accuracy')
    plt.show()

    # * tf2.1之前：当数据集很大的时候，不能一次性将数据集载入模型，所以使用model.fit_generator；而小数据集才使用model.fit
    # tf2.1之后，model.fit已经兼容了model.fit_generator

   
    history = model.fit_generator(generator=train_data_gen,
                                  steps_per_epoch=total_train // batch_size,
                                  epochs=epochs,
                                  validation_data=val_data_gen,
                                  validation_steps=total_val // batch_size,
                                  callbacks=callbacks)


"""     # 下面的使用的是 低级的API 
    # using keras low level api for training
    # 定义loss、优化器、训练过程、测试过程
    loss_object = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005)
    
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.CategoricalAccuracy(name='train_accuracy')
    
    test_loss = tf.keras.metrics.Mean(name='test_loss')
    test_accuracy = tf.keras.metrics.CategoricalAccuracy(name='test_accuracy')
    
    
    @tf.function
    def train_step(images, labels):
        with tf.GradientTape() as tape:
            predictions = model(images, training=True)
            loss = loss_object(labels, predictions)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    
        train_loss(loss)
        train_accuracy(labels, predictions)
    
    
    @tf.function
    def test_step(images, labels):
        predictions = model(images, training=False)
        t_loss = loss_object(labels, predictions)
    
        test_loss(t_loss)
        test_accuracy(labels, predictions)
    
    
    best_test_loss = float('inf')
    for epoch in range(1, epochs+1):
        train_loss.reset_states()        # clear history info
        train_accuracy.reset_states()    # clear history info
        test_loss.reset_states()         # clear history info
        test_accuracy.reset_states()     # clear history info
        for step in range(total_train // batch_size):
            images, labels = next(train_data_gen)
            train_step(images, labels)
    
        for step in range(total_val // batch_size):
            test_images, test_labels = next(val_data_gen)
            test_step(test_images, test_labels)
    
        template = 'Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}'
        print(template.format(epoch,
                              train_loss.result(),
                              train_accuracy.result() * 100,
                              test_loss.result(),
                              test_accuracy.result() * 100))
        if test_loss.result() < best_test_loss:
           model.save_weights("./save_weights/myAlex.ckpt", save_format='tf')
           # 模型保存 model.save_weights 只保存权重，保存的格式是ckpt（官方的） """


if __name__ == '__main__':
    main()
