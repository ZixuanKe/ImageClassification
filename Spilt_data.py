#coding: utf-8
from PIL import Image
import numpy as np;
import random

data_train = np.empty((3400, 1, 15, 15), dtype="float32")
label_train = np.empty((3400,), dtype="uint8")
data_test = np.empty((229160, 1, 15, 15), dtype="float32")
label_test = np.empty((229160,), dtype="uint8")


def load_data():
    import glob
    num_train = 0
    num_test = 0
    print "begin: "
    # 获取指定目录下的所有图片
    for f in ['0','1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22','23','24','25','26','27','28','29',
                   '30','31','32','33']:
        print "f: " + f
        location = "C:\Users\IBM_ADMIN\PycharmProjects\ImageClassification\image\\" + f + "\*.bmp"
        print "num_train: " + str(num_train)
        image_file = glob.glob(location)
        #返回符合匹配的所有文件的list

        image_file_train = random.sample(image_file, 100)  # 选取其中100个作为训练图像
        image_file_test = [item for item in image_file if item not in image_file_train]  #剩余的为测试数据集
        print  "test: " ,len(image_file_test)
        print "train: ",len(image_file_train)

        #test
        for i in range(len(image_file_test)):
            img = Image.open(image_file_test[i])
            arr = np.asarray(img,dtype="float32")
            data_test[num_test,:,:,:] = arr
            label_test[num_test] = f
            num_test += 1                           #逐个标签输入test

        #train
        for i in range(len(image_file_train)):
            img = Image.open(image_file_train[i])
            arr = np.asarray(img,dtype="float32")
            data_train[num_train,:,:,:] = arr
            label_train[num_train] = f
            num_train += 1                      #逐个标签输入train

    np.save("data_train.npy",data_train)
    np.save("label_train.npy",label_train)
    np.save("data_test.npy",data_test)
    np.save("label_test.npy",label_test)



if __name__ == "__main__":
    load_data()