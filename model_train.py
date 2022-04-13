
from keras import Sequential
from keras.layers import *
from keras import optimizers
import numpy as np
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt


def data_pre(data_path, class_num):
    '''
     读取训练集和验证集
    :param data_path:
    :return:
    '''
    x_train = []
    y_train = []
    text = open(data_path + 'train.txt', 'r', encoding='utf-8-sig').read().split('\n')
    for i in text:
        if i:
            wenjian = i.split('	')[0]
            leibie = i.split('	')[1]
            x = np.load( wenjian).reshape((256, 256, 1))
            x_train.append(x)
            y_train.append((int(leibie)))

    x_train = np.array(x_train)  # 将数据从list列表转化为numpy的形式
    y_train = np.array(y_train)  # 将数据从list列表转化为numpy的形式
    y_train = np_utils.to_categorical(y_train, class_num)  # 将数据标签 转化为one-hot形式
    '''
         0               1   0
         1               0   1

         0               1   0   0   0   0   0
         1               0   1   0   0   0   0
         2               0   0   1   0   0   0
         3               0   0   0   1   0   0
         4               0   0   0   0   1   0
         5               0   0   0   0   0   1

    '''
    # 添加验证集
    x_dev = []
    y_dev = []
    text = open(data_path + 'dev.txt', 'r', encoding='utf-8-sig').read().split('\n')
    for i in text:
        if i:
            wenjian = i.split('	')[0]
            leibie = i.split('	')[1]
            x = np.load(wenjian).reshape((256, 256, 1))
            x_dev.append(x)
            y_dev.append((int(leibie)))
    x_dev = np.array(x_dev)
    y_dev = np.array(y_dev)
    y_dev = np_utils.to_categorical(y_dev, class_num)

    return x_train, y_train, x_dev, y_dev


def train_model(x_train, y_train, x_dev, y_dev, batch_size, epochs, class_num, savepath):
    model = Sequential()  # keras框架  序列化的建立模型的方法  另一个函数式
    model.add(Conv2D(filters=16, kernel_size=(5, 5), padding='same', input_shape=(256, 256, 1), activation='relu'))
    # filters 卷积核的个数  kernel_size  每个卷积核的尺寸  strides  步进 默认与kernel_size一样， padding  输出的图的尺寸  是否要与输入保持一致  input_shape  输入的尺寸  activation 激活函数
    model.add(Conv2D(filters=16, kernel_size=(5, 5), padding='same', activation='relu'))
    # 第二个卷积
    model.add(MaxPool2D(pool_size=(2, 2)))
    # 池化层  最大池化  pool_size 就是池化格子的尺寸

    model.add(Dropout(0.25))
    # 防止过拟合的  断开一定比例的神经元之间的连接   过拟合 就是 在训练集上表现很好 但是测试集表现不好
    model.add(Conv2D(filters=32, kernel_size=(5, 5), padding='same', activation='relu'))
    model.add(Conv2D(filters=32, kernel_size=(5, 5), padding='same', activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))  # [a,b,c]

    model.add(Reshape((32, -1)))  # 32* x =a*b*c
    # Reshape层  在该层之前  都是三维的数据（不考虑batch_size的情况下），变为了2维数据，
    model.add(Dense(100, activation='relu'))
    # 全连接层   激活函数  relu   输出的最后一个维度变为100  32,100
    # y = x*k+b     x的维度是[5, x]   k 权重的维度是 [x,100]   b bias 偏置 [5,100]   y 维度  就是[5,100]

    model.add(Flatten())

    # Flatten  扁平化   [,32*100]
    model.add(Dense(class_num, activation='softmax'))  # 【1， 6】
    #  第二个全连接层   激活函数 softmax      [1,2]
    '''
          softmax之前    1   2
          softmax之后    e^1/(e^1+e^2)  + e^2/(e^1+e^2)   =  1     实际输出
          标签           0                      1                  理论输出
    '''
    model.summary()
    # 总结
    opt = optimizers.adam(lr=0.0001)
    # 优化器  adam优化器   学习率   0.0001
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    #   损失函数  categorical_crossentropy  交叉熵损失函数
    model_checkpoint = ModelCheckpoint(
        savepath,
        monitor='val_acc',
        verbose=1,
        save_best_only=True)
    # 最终保存的模型是  在验证集上表现最好的模型
    history = model.fit(x_train, y_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        validation_data=(x_dev, y_dev),
                        shuffle=True,
                        callbacks=[model_checkpoint])
    # 训练  每次放入16组数组   训练10轮
    # 模型训练的轮数就是epochs ，训练10轮，训练集每次训练都能保存一个模型，训练10轮就是保存10个模型文件
    # 这里面最好的模型的意思就是说  训练训练的10个模型，每个都要用验证集来测试 验证集的准确率 val_acc
    # val_acc就是监控的指标，相当于10个保存好的模型，哪一个在测试集的val_acc上的准确率最高，那么久保存哪个模型
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Ver'], loc='upper right')
    plt.savefig("pic/Loss_cnn_mfcc.png")
    plt.show()

    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('Model acc')
    plt.ylabel('Acc')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Ver'], loc='upper right')
    plt.savefig("pic/Acc_cnn_mfcc.png")
    plt.show()


if __name__ == '__main__':
    x_train, y_train, x_dev, y_dev = data_pre('file/', class_num=2)
    train_model(x_train, y_train, x_dev, y_dev,
                batch_size=16,
                epochs=10,
                class_num=2,
                savepath="model/best_model_cnn.h5")