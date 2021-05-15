from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras import Model


# ! tf搭建模型的方式，这里采用的是 Model Subclassing API，类似于pytorch的搭建风格

class MyModel(Model):                                       # 继承于tensorflow.keras.Model父类
    def __init__(self):                                     # & __init__定义模块
        super(MyModel, self).__init__()
        self.conv1 = Conv2D(32, 3, activation='relu')       # ^ (filter, kernel_size, stride……    )；默认valid padding
                                                            # * 自己跳转到定义看看Conv2D  
        self.flatten = Flatten()                            # 展平参数
        self.d1 = Dense(128, activation='relu')             # dense的units=128，表示的是输出的节点个数    
        self.d2 = Dense(10, activation='softmax')

        # ^ pytorch中的定义需要写出上层的信息（比如上一层的输入是多少）；
        # ! 但是tf中只需要写本层的信息的就可以了，tensorflow可以自己推理；参数要少很多

    def call(self, x, **kwargs):                            # & call定义网络正向传播的过程
        x = self.conv1(x)      # input[batch, 28, 28, 1] output[batch, 26, 26, 32]
                               # 默认是valid padding； N = (W-F+1)/S，所以是28-3+1=26
        x = self.flatten(x)    # output [batch, 21632]； 26*26*32=21632 展为一维的项链格式
        x = self.d1(x)         # output [batch, 128]
        return self.d2(x)      # output [batch, 10]； 最后是十类别

