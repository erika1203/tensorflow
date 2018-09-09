'''
2018-09-04
一个简单的自编码器例子
tf02
'''

import numpy as np
import sklearn.preprocessing as prep
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

def xavier_init(fan_in,fan_out,constant=1):
    #创建均匀分布
    low=-constant*np.sqrt(6.0/(fan_in+fan_out))
    high=constant*np.sqrt(6.0/(fan_in+fan_out))
    return tf.random_uniform((fan_in,fan_out),minval=low,maxval=high,dtype=tf.float32)

class AdditiveGaussianNoiseAutoencoder(object):
    #定义一个去噪自编码器
    def __init__(self,n_input,n_hidden,transfer_function=tf.nn.softplus,optimizer=tf.train.AdamOptimizer(),scale=0.1):
        self.n_input=n_input    #输入变量数
        self.n_hidden=n_hidden  #隐含层节点数
        self.transfer=transfer_function #隐含层激活函数，优化器默认为Adam
        self.scale=tf.placeholder(tf.float32)   #高斯噪声系数，默认为0.1
        self.training_scale=scale
        network_weights=self._initialize_weights()  #参数初始化，在下面定义
        self.weights=network_weights        #参数
        self.x=tf.placeholder(tf.float32,[None,self.n_input])   #输入，占位符
        #一个能提取特征的隐藏层，输入x加上噪声，然后与权重w1相乘，加上偏置b1，然后激活处理
        self.hidden=self.transfer(tf.add(tf.matmul(
                        self.x+scale*tf.random_normal((n_input,)),
                        self.weights['w1']),self.weights['b1']))
        #在输出层进行数据复原重建，不需激活函数，hidden乘输出层权重w2，再加上输出层偏置b2
        self.reconstruction=tf.add(tf.matmul(self.hidden,self.weights['w2']),self.weights['b2'])
        #定义损失函数，用平方误差
        self.cost=0.5*tf.reduce_sum(tf.pow(tf.subtract(self.reconstruction,self.x),2.0))
        #定义优化器（最小化损失函数），优化器默认为Adam
        self.optimizer=optimizer.minimize(self.cost)
        # 创建Session，初始化全部参数
        init=tf.global_variables_initializer()
        self.sess=tf.Session()
        self.sess.run(init)

    def _initialize_weights(self):
        #参数字典，初始化
        all_weights=dict()
        all_weights['w1']=tf.Variable(xavier_init(self.n_input,self.n_hidden))
        all_weights['b1']=tf.Variable(tf.zeros([self.n_hidden],dtype=tf.float32))
        all_weights['w2']=tf.Variable(tf.zeros([self.n_hidden,self.n_input],dtype=tf.float32))
        all_weights['b2']=tf.Variable(tf.zeros([self.n_input],dtype=tf.float32))
        return all_weights

    def partial_fit(self,X):
        #批量训练，返回损失
        cost,opt=self.sess.run((self.cost,self.optimizer),
                               feed_dict={self.x:X,self.scale:self.training_scale})
        return cost

    def calc_total_cost(self,X):
        #只求损失不会触发训练，用于测试
        return self.sess.run(self.cost,feed_dict={self.x:X,self.scale:self.training_scale})

    def transform(self,X):
        #隐含层的输出结果，学习出数据中抽象后的高阶特征
        return self.sess.run(self.hidden,feed_dict={self.x:X,self.scale:self.training_scale})

    def generate(self,hidden=None):
        #将隐含层输出作为输入，通过重建层恢复为原始数据
        if hidden is None:
            hidden=np.random.normal(size=self.weights['b1'])
        return self.sess.run(self.reconstruction,feed_dict={self.hidden:hidden})

    def reconstruct(self,X):
        #包括transform和generate，输入原数据，输出复原数据，整体运行一遍复原过程
        return self.sess.run(self.reconstruction,
                             feed_dict={self.x:X,self.scale:self.training_scale})

    def getWeights(self):
        #获取权重w1
        return self.sess.run(self.weights['w1'])

    def getBiases(self):
        #获取偏置系数b1
        return self.sess.run(self.weights['b1'])

mnist=input_data.read_data_sets('MNIST_data',one_hot=True)

def standard_scale(X_train,X_test):
    #对训练，测试数据进行标准化处理，使数据均值为0，标准差为1
    preprocessor=prep.StandardScaler().fit(X_train)
    X_train=preprocessor.transform(X_train)
    X_test=preprocessor.transform(X_test)
    return X_train,X_test

def get_random_block_from_data(data,batch_size):
    #随机抽样一个batch，不放回
    start_index=np.random.randint(0,len(data)-batch_size)
    return data[start_index:(start_index+batch_size)]

X_train,X_test=standard_scale(mnist.train.images,mnist.test.images)
n_samples=int(mnist.train.num_examples)     #总训练样本数
training_epochs=20  #训练最大轮数
batch_size=128      #一批训练样本
display_step=1  #每隔一轮显示一次损失

#创建一个AGN自编码器实例
autoencoder=AdditiveGaussianNoiseAutoencoder(n_input=784,
                                             n_hidden=200,
                                             transfer_function=tf.nn.softplus,
                                             optimizer=tf.train.AdamOptimizer(learning_rate=0.001),
                                             scale=0.01)
for epoch in range(training_epochs):
    #每一轮训练将均差设为0
    avg_cost=0.
    #分批数
    total_batch=int(n_samples/batch_size)
    for i in range(total_batch):
        #每一批进行随机抽取，部分训练并获得误差
        batch_xs=get_random_block_from_data(X_train,batch_size)
        cost=autoencoder.partial_fit(batch_xs)
        #计算均差
        avg_cost+=cost/n_samples*batch_size
    #打印轮数和均差
    if epoch % display_step==0:
        print('Epoch:','%04d' %(epoch+1),
              'cost=','{:.9f}'.format(avg_cost))
    print('Total cost:'+str(autoencoder.calc_total_cost(X_test)))




