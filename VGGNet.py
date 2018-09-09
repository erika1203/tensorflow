'''
2018-09-06
深度学习TensorFlow
VGGNet-16 version D
'''

from datetime import datetime
import math
import time
import tensorflow as tf

def conv_op(input_op,name,kh,kw,n_out,dh,dw,p):
    '''
    定义一个卷积层
    :param input_op: 输入tensor
    :param name: 这一层的名称
    :param kh: 卷积核的高
    :param kw: 卷积核的宽
    :param n_out: 卷积核数量（即输出通道数）
    :param dh: 步长的高
    :param dw: 步长的宽
    :param p: 参数列表
    :return:
    '''
    n_in=input_op.get_shape()[-1].value
    with tf.name_scope(name) as scope:
        #设置卷积核参数
        kernel=tf.get_variable(scope+'w',
                               shape=[kh,kw,n_in,n_out],dtype=tf.float32,
                               #参数初始化
                               initializer=tf.contrib.layers.xavier_initializer_conv2d())
    conv=tf.nn.conv2d(input_op,kernel,(1,dh,dw,1),padding='SAME')
    #将偏置赋值为0
    bias_init_val=tf.constant(0.0,shape=[n_out],dtype=tf.float32)
    #再转成可训练的参数
    biases=tf.Variable(bias_init_val,trainable=True,name='b')
    #conv与偏置相加，再进行非线性处理
    z=tf.nn.bias_add(conv,biases)
    activation=tf.nn.relu(z,name=scope)
    #卷积核参数和偏置添加到参数列表中
    p+=[kernel,biases]
    return activation

def fc_op(input_op,name,n_out,p):
    #定义一个全连接层
    n_in=input_op.get_shape()[-1].value
    with tf.name_scope(name) as scope:
        kernel=tf.get_variable(scope+'w',
                               shape=[n_in,n_out],dtype=tf.float32,
                               initializer=tf.contrib.layers.xavier_initializer())
        biases=tf.Variable(tf.constant(0.1,shape=[n_out],
                                       dtype=tf.float32),name='b')
        activation=tf.nn.relu_layer(input_op,kernel,biases,name=scope)
        p+=[kernel,biases]
        return activation

def mpool_op(input_op,name,kh,kw,dh,dw):
    #定义一个最大池化层
    return tf.nn.max_pool(input_op,
                          ksize=[1,kh,kw,1],    #池化尺寸
                          strides=[1,dh,dw,1],  #步长
                          padding='SAME',
                          name=name)

def inference_op(input_op,keep_prob):
    #创建VGGNet-16的网络结构
    p=[]    #参数列表
    #第1段卷积网络，2个卷积层，1个最大池化层，输出112*112*64
    conv1_1=conv_op(input_op,name='conv1_1',kh=3,kw=3,n_out=64,dh=1,dw=1,p=p)
    conv1_2=conv_op(conv1_1,name='conv1_2', kh=3, kw=3, n_out=64, dh=1, dw=1, p=p)
    pool1=mpool_op(conv1_2,name='pool1',kh=2,kw=2,dw=2,dh=2)

    #第2段卷积网络，2个卷积层，1个最大池化层，输出56*56*128
    conv2_1=conv_op(pool1,name='conv2_1',kh=3,kw=3,n_out=128,dh=1,dw=1,p=p)
    conv2_2=conv_op(conv2_1,name='conv2_2', kh=3, kw=3, n_out=128, dh=1, dw=1, p=p)
    pool2=mpool_op(conv2_2,name='pool2',kh=2,kw=2,dw=2,dh=2)

    #第3段卷积网络，3个卷积层，1个最大池化层，输出28*28*256
    conv3_1=conv_op(pool2,name='conv3_1',kh=3,kw=3,n_out=256,dh=1,dw=1,p=p)
    conv3_2=conv_op(conv3_1,name='conv3_2', kh=3, kw=3, n_out=256, dh=1, dw=1, p=p)
    conv3_3=conv_op(conv3_2,name='conv3_3', kh=3, kw=3, n_out=256, dh=1, dw=1, p=p)
    pool3=mpool_op(conv3_3,name='pool3',kh=2,kw=2,dw=2,dh=2)

    #第4段卷积网络，3个卷积层，1个最大池化层，输出14*14*512
    conv4_1 = conv_op(pool3, name='conv4_1', kh=3, kw=3, n_out=512, dh=1, dw=1, p=p)
    conv4_2 = conv_op(conv4_1, name='conv4_2', kh=3, kw=3, n_out=512, dh=1, dw=1, p=p)
    conv4_3 = conv_op(conv4_2, name='conv4_3', kh=3, kw=3, n_out=512, dh=1, dw=1, p=p)
    pool4 = mpool_op(conv4_3, name='pool4', kh=2, kw=2, dw=2, dh=2)

    #第5段卷积网络,3个卷积层，1个最大池化层，输出7*7*512
    conv5_1 = conv_op(pool4, name='conv5_1', kh=3, kw=3, n_out=512, dh=1, dw=1, p=p)
    conv5_2 = conv_op(conv5_1, name='conv5_2', kh=3, kw=3, n_out=512, dh=1, dw=1, p=p)
    conv5_3 = conv_op(conv5_2, name='conv5_3', kh=3, kw=3, n_out=512, dh=1, dw=1, p=p)
    pool5 = mpool_op(conv5_3, name='pool5', kh=2, kw=2, dw=2, dh=2)

    #将第5段卷积输出结果扁平化为7*7*512=25088的一维向量
    shp=pool5.get_shape()
    flattened_shape=shp[1].value*shp[2].value*shp[3].value
    resh1=tf.reshape(pool5,[-1,flattened_shape],name='resh1')

    #连接一个全连接层，隐含层节点4096
    fc6=fc_op(resh1,name='fc6',n_out=4096,p=p)
    #连接一个Dropout层，训练保留率0.5，预测保留率1.0
    fc6_drop=tf.nn.dropout(fc6,keep_prob,name='fc6_drop')
    #又一个全连接层和Dropout层
    fc7 = fc_op(fc6_drop, name='fc7', n_out=4096, p=p)
    fc7_drop = tf.nn.dropout(fc7, keep_prob, name='fc7_drop')

    #最后连接一个1000输出的全连接层，使用softmax得到分类输出概率
    fc8=fc_op(fc7_drop,name='fc8',n_out=1000,p=p)
    softmax=tf.nn.softmax(fc8)
    predictions=tf.argmax(softmax,1)
    return predictions,softmax,fc8,p

def time_tensorflow_run(session,target,feed,info_string):
    #计算执行时间
    num_steps_burn_in=10
    total_duration=0.
    total_duration_squared=0.
    for i in range(num_batches+num_steps_burn_in):
        start_time=time.time()
        _=session.run(target,feed_dict=feed)
        duration=time.time()-start_time
        if i >= num_steps_burn_in:
            if not i % 10:
                print('%s : step %d,duration= %.3f' %
                      (datetime.now(),i-num_steps_burn_in,duration))
            total_duration+=duration
            total_duration_squared+=duration*duration
    mn=total_duration/num_batches   #每批数据所耗费的时间
    vr=total_duration_squared/num_batches-mn*mn
    sd=math.sqrt(vr)
    print('%s : %s across %d steps, %.3f +/- %.3f sec/batch' %
              (datetime.now(),info_string,num_batches,mn,sd))

def run_benchmark():
    #主函数
    with tf.Graph().as_default():
        #生成224*224的随机图片
        image_size=224
        #生成标准差为0.1的正态分布随机数
        images=tf.Variable(tf.random_normal([batch_size,image_size,image_size,3],
                                            dtype=tf.float32,stddev=1e-1))
        #创建keep_prob的placeholder
        keep_prob=tf.placeholder(tf.float32)
        #创建一个VGGNet实例
        predictions,softmax,fc8,p=inference_op(images,keep_prob)
        #创建session并初始化全局参数
        init=tf.global_variables_initializer()
        sess=tf.Session()
        sess.run(init)

        time_tensorflow_run(sess,predictions,{keep_prob:1.0},'Forward')
        objective=tf.nn.l2_loss(fc8)
        grad=tf.gradients(objective,p)
        time_tensorflow_run(sess,grad,{keep_prob:0.5},'Forward-backward')


batch_size=32
num_batches=100
run_benchmark()





