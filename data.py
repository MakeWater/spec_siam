# -*- coding: utf-8 -*-
from keras.datasets import mnist
import numpy as np
from spec import predict_data

def get_data():
    # mnist = input_data.read_data_sets('MNIST_data', one_hot= False)

    (data1, label_train),(data2, label_test) = mnist.load_data()
    all_unlabeled_data = np.concatenate((data1, data2), axis= 0)
    all_label= np.concatenate((label_train, label_test),axis= 0)

    return all_unlabeled_data, all_label
. 
def get_pairs(data):
    # ret= {}
    # data= get_unlabeled_data()
    params= {'n_clsuters':10, 'n_nbrs':2}
    y_train = predict_data(data,params['n_clusters'], params['n_nbrs']) # 用谱聚类方法获取这七万张图片每张图片的标签。这些标签还将用于孪生网络
    (x_train_labeled, y_train_labeled) = split_data(data, y_train)

    # get pairs from labeled data
    class_indices = [np.where(y_train_labeled == i)[0] for i in range(params['n_clusters'])]
    pairs_train, pairs_label = create_pairs(x_train_labeled, class_indices)
    # ret['pairs_train']= pairs_train
    # ret['pairs_label']= pairs_train_labeled #分别是数据对数组和数据对的标签的数组。
    return pairs_train, pairs_label, y_train

# 获取各个类别的样本下标，即按照类别来对样本集分组：
# class_indices = [np.where(y_pred_label == i)[0] for i in range(cluster_num)]

def split_data(data, label):
    '''
    把数据x、标签y打包为元组。为了下一步进行带标签数据的配对。
    x: 深度为2的数据
    y： 深度为1的标签
    '''
    assert len(data) == len(label)
    n = len(data)
    p= np.arange(n) # p的作用相当于设置显式的下标
    ret_x_y = []
    # 将数据和对应的标签组合起来
    for i in range(n)
        x_ = data[i]
        y_ = label[i]
        ret_x_y.append((x_,y_))
    return tuple(ret_x_y)


 
def create_pairs(x, class_indices):
    '''Positive and negative pair creation.
    Alternates between positive and negative pairs.
    '''
    pairs = [] #一会儿一对对的样本要放在这里
    labels = [] # 这是对的样本，但是不是数据的样本，，，

    # 将n设置为所有类别样本数目之最小值，可以保证对所有类别而言，
    # 生成的正样本数目和负样本数目都是一样的，从而保证整个训练集的类别均衡。
    #-1是因为在循环中需要访问[i+1]，这是为了保证不超出范围。

    n = min([len(class_indices[d]) for d in range(10)]) - 1
    for d in range(10):
        #对第d类抽取正负样本
        for i in range(n):
            # 遍历d类的样本，取临近的两个样本为正样本对
            z1, z2 = class_indices[d][i], class_indices[d][i+1]
            pairs += [[x[z1], x[z2]]]
            # randrange会产生1~9之间的随机数，含1和9
            inc = random.randrange(1, 10)
            # (d+inc)%10一定不是d，用来保证负样本对的图片绝不会来自同一个类
            dn = (d + inc) % 10
            # 在d类和dn类中分别取i样本构成负样本对
            z1, z2 = class_indices[d][i], class_indices[dn][i]
            pairs += [[x[z1], x[z2]]]
            # 添加正负样本标签
            labels += [1, 0]
    # x.shape[1:] 表示的是数据x的一行有多少列，对于mnist数据集来说就是784列。
    pairs= np.array(pairs).reshape((len(pairs),2))
    labels= np.array(labels)
    return pairs, labels

def getCenters(data, y_pred_label):
    '''
    data is all_data
    y_pred_label is label predicted by spectral clustering.
    '''
    centers= []
    for i in range(max(y_pred_label)+ 1):
        points_list= np.where(y_pred_label == i)[0].tolist()
        centers.append(np.average(data[points_list], axis= 0))

    return centers