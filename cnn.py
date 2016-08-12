#coding:utf-8

import os
from PIL import Image
import numpy as np
from keras.utils import np_utils, generic_utils
from keras.models import Sequential, Model
from keras.layers import Embedding, Convolution2D, Input, Activation, MaxPooling2D, Reshape, Dropout, Dense, \
    Flatten, Merge
from keras.optimizers import SGD
from keras.models import Sequential
from keras.layers import Convolution2D, ZeroPadding2D, MaxPooling2D
from keras.models import model_from_json
import numpy as np;
np.random.seed(1337)  # for reproducibility
import random

# filter 滤波  即卷积核

nb_pool = [2,2]
nb_classes = 34

layer1 = 32
hidden1 = 700
hidden2 = 500

channel = 1
length = 15
width = 15

lr=0.001
decay=1e-6
momentum=0.9

# layer_name = 'conv1_1'
layer_name = 'dense_3'

step = 0.01
filter_index = 10


def output(output_model,input_img):
    layer_dict = dict([(layer.name, layer) for layer in output_model.layers])

    #loss function:
    from keras import backend as K  #封装底层操作

     # can be any integer from 0 to 511, as there are 512 filters in that layer

    # build a loss function that maximizes the activation
    # of the nth filter of the layer considered
    layer_output = layer_dict[layer_name].output

    # loss = K.mean(layer_output[:, filter_index, :, :])   #求向量均值 损失函数 二次平方和误差
    loss = K.mean(layer_output[:,  filter_index])

    # compute the gradient of the input picture wrt this loss
    grads = K.gradients(loss, input_img)[0]     #返回梯度列表（求导操作，[0]即最小梯度）

    # normalization trick: we normalize the gradient
    grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5)   #开始进行梯度下降

    # this function returns the loss and grads given the input picture
    iterate = K.function([input_img], [loss, grads])    #实例化向量

    # gradient_decent(width, length,iterate,0.01))
    import numpy as np

    # we start from a gray image with some noise
    # input_img_data = np.random.random((1, 1, 15, 15)) * 20 + 128.   #随机初始化
    input_img_data = np.random.random((15, 15)) * 20 + 128.   #随机初始化

    # run gradient ascent for 20 steps
    for i in range(20):
        loss_value, grads_value = iterate([input_img_data])
        input_img_data += grads_value * step

    #迭代20次

    from scipy.misc import imsave


    # util function to convert a tensor into a valid image
    def deprocess_image(x): #提取结果
        # normalize tensor: center on 0., ensure std is 0.1
        x -= x.mean()
        x /= (x.std() + 1e-5)
        x *= 0.1

        # clip to [0, 1]
        x += 0.5
        x = np.clip(x, 0, 1)

        # convert to RGB array
        x *= 255
        print x.shape
        # x = x.transpose((1, 2, 0))

        x = np.clip(x, 0, 255).astype('uint8')  #剪裁
        return x


    img = input_img_data[0]
    img = deprocess_image(img)
    # img = img.reshape(15,15)
    print img
    img = img.reshape(15,1)# newX=numpy.resize(X,(len(X),15*15))

    # img = np.array(img,np.int32)
    imsave('%s_filter_%d.png' % (layer_name, filter_index), img)

def load_data(data,label):
    rdata = np.load(data)
    rlabel = np.load(label)
    return rdata,rlabel


if __name__ == '__main__':
    train_data,train_label = load_data('data_train.npy','label_train.npy')
    test_data,test_label = load_data('data_test.npy','label_test.npy')

    label_train = train_label
    label_train = np_utils.to_categorical(label_train, 34)  # 必须使用固定格式表示标签
    label_test = test_label
    label_test = np_utils.to_categorical(label_test, 34)  # 必须使用固定格式表示标签


    layer1_model1=Sequential()
    layer1_model1.add(Convolution2D(layer1, 3, 3,
                            border_mode='valid',
                            input_shape=(channel, length, width),name='conv1_1')

                      )
    input_img_conv = layer1_model1.layers[-1].input

    layer1_model1.add(Activation('tanh'))
    layer1_model1.add(MaxPooling2D(pool_size=(nb_pool[0], nb_pool[1])))


    layer1_model2=Sequential()
    layer1_model2.add(Convolution2D(layer1, 5, 5,
                            border_mode='valid',
                            input_shape=(channel, length, width),name='conv1_2')
                      )
    layer1_model2.add(Activation('tanh'))
    layer1_model2.add(MaxPooling2D(pool_size=(nb_pool[0], nb_pool[1])))

    #16*20*1
    layer1_model3=Sequential()
    layer1_model3.add(Convolution2D(layer1, 7, 7,
                            border_mode='valid',
                            input_shape=(channel, length, width),name='conv1_3'))
    layer1_model3.add(Activation('tanh'))
    layer1_model3.add(MaxPooling2D(pool_size=(nb_pool[0], nb_pool[1])))

    layer1_model1.add(Flatten())
    layer1_model2.add(Flatten())
    layer1_model3.add(Flatten())


    model = Sequential()

    model.add(Merge([layer1_model2,layer1_model1,layer1_model3], mode='concat',concat_axis=1))#merge

    # model.add(Flatten())  # 平铺

    model.add(Dense(hidden1))  # Full connection 1:  1000
    model.add(Activation('tanh'))
    model.add(Dropout(0.5))

    model.add(Dense(hidden2))  # Full connection 1:  1000
    model.add(Activation('tanh'))
    model.add(Dropout(0.5))

    model.add(Dense(34))  # Full connection 1:  1000

    model.add(Activation('softmax'))
    model.add(Dropout(0.5))


    input_img_dense = model.layers[-3].input

    output(model,input_img_dense)

    #
    # sgd = SGD(lr=lr, decay=decay, momentum=momentum, nesterov=True)
    # model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=["accuracy"])
    #
    # model.fit([train_data,train_data,train_data], label_train, batch_size=32, nb_epoch=30,
    #                           shuffle=True, verbose=1, validation_split=0)
    #
    #
    # print '训练准确率：'
    # print model.metrics_names
    # print model.evaluate([train_data,train_data,train_data], label_train, show_accuracy=True)
    #
    # print '测试准确率：'
    # print model.metrics_names
    # print model.evaluate([test_data,test_data,test_data], label_test, show_accuracy=True)
    #
    #


# newX=numpy.resize(X,(len(X),15*15))
# 如何输入RF
# 多维变二维