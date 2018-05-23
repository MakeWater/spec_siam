# -*- coding: utf-8 -*-
import tensorflow as tf
from keras.layers import Input, Dense
from keras.models import Model, Sequential
import numpy as np 
from keras.optimizers import SGD, RMSprop
from keras import backend as K
from keras.layers import Input, Lambda
from keras.models import Model
from utility import euclidean_distance, eucl_dist_output_shape
from layer import make_layer_list, stack_layers
from data import get_unlabeled_data


def creat_siamese_network(data，params):
    '''
    base network to be shared(eq to feature extraction).
    '''
    x= get_unlabeled_data()
    pairs_train, pairs_label = data['pairs_train'], data['pairs_label']
    input_shape= x.shape[1:]

    siamese_inputs= {'A':Input(shape= input_shape)
                     'B': Input(shape= input_shape)
                    }
    # y_ 是谱聚类预测的标签。
    y_ = predict_data(cluster_num= params['n_clusters'])
    train_gen_ = train_gen(pairs_train, pairs_label, params['siam_batch_size'])
    # y 是对标签。
    y = train_gen_['label_y']

    layers= []
    layers += make_layer_list(params['arch'], 'siamese', params.get('siam_reg')) # 这里params.get('siam_reg')其实等于“None”
        '''
        layers=[{'l2_reg': reg，'type': 'relu', 'size': 1024，'name':'siamese_0'},{'name': 'siamese_dropout_0', 'rate': 0.05, 'type': 'Dropout'},
                {'l2_reg': reg，'type': 'relu', 'size': 1024，'name':'siamese_1'},{'name': 'siamese_dropout_0', 'rate': 0.05, 'type': 'Dropout'},
                {'l2_reg': reg，'type': 'relu', 'size': 512，'name':'siamese_2'},{'name': 'siamese_dropout_0', 'rate': 0.05, 'type': 'Dropout'},
                {'l2_reg': reg，'type': 'relu', 'size': 10，'name':'siamese_3'},{'name': 'siamese_dropout_0', 'rate': 0.05, 'type': 'Dropout'}
              ]
        '''
    # 画出这一步后的网络结构：
    siamese_outputs = stack_layers(siamese_inputs, layers) # 这一步创建层模型，并且最终得到的输出还是dropout 层，不是softmax层得到的输出结果，
    # 注意上面的“siamese_inputs”， 这个里面的inputs怎么传进去的以及里面的参数分别是怎么被使用的？注意创建/定义工具与实例化一个工具后再使用的区别。
    # siamese_outputs={'A':InputA的输出，'B':InputB的输出}

    distance = Lambda(euclidean_distance, output_shape= eucl_dist_output_shape)([siamese_outputs['A'], siamese_outputs['B']])
    # 下面建立了一个从输入到输出的模型：
    siamese_net_distance = Model([siamese_inputs['A'], siamese_intputs['B']], distance)

    loss = contrastive_loss(distance, y) # 实例化一个自定义loss函数后才能传入配置中。
    # 配置网络参数：
    # 设想：Model([siamese_outputs['A'], siamese_outputs['B']], distance).compile(optimizer= 'RMSprop',loss= contrastive_loss)
    siamese_net_distance.compile(optimizer= 'RMSprop',loss= loss) # 这个loss 函数怎样使用网络传出的数据？，包括两个embaddings 和一个标签。

    
    

    # create handler for early stopping and learning rate scheduling 为提前停止和学习率变化设置learning handler处理程序
    siam_lh = LearningHandler(
            lr=params['siam_lr'],# 设置学习率
            drop=params['siam_drop'],  #学习率递减速度
            lr_tensor=siamese_net_distance.optimizer.lr,
            patience=params['siam_patience'])
    #初始化训练数据：
    train_gen_ = train_gen(pairs_train, pairs_label, params['siam_batch_size'])
    y = train_gen_['label_y']
    # compute the steps per epoch
    steps_per_epoch = int(len(pairs_train) / params['siam_batch_size'])
    
    # 返回一个history对象
    hist = siamese_net_distance.fit_generator(train_gen_['A'],train_gen_['B'],y, 
                                                  epochs=params['siam_ne'], 
                                                  validation_data=None,
                                                  steps_per_epoch=steps_per_epoch,
                                                  callbacks=[siam_lh])
 
    
    # siamese_net_distance.fit(train_gen_)   #开始训练
# 定义孪生网络的损失函数
def contrastive_loss(distance, y):
    margin = params['margin']
    tmp = y*tf.square(distance)
    tmp2 = (1-y)*tf.square(tf.maximum((margin-d), 0))
    return tf.reduce_mean(tmp + tmp2)/2
    
def contrastive_loss(y_true, y_pred):
    '''Contrastive loss from Hadsell-et-al.'06
    http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    '''
    margin = 1
    return K.mean(y_true * K.square(y_pred) +
                (1 - y_true) * K.square(K.maximum(margin - y_pred, 0)))