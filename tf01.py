'''
2018-09-04
mnist数字识别简单例子
tf01
'''

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

def MnistTest():
    #导入数据
    mnist=input_data.read_data_sets('MNIST_data/',one_hot=True)
    print(mnist.train.images.shape,mnist.train.labels.shape)
    print(mnist.test.images.shape,mnist.test.labels.shape)
    print(mnist.validation.images.shape,mnist.validation.labels.shape)
    #实现softmax回归
    sess=tf.InteractiveSession()
    x=tf.placeholder(tf.float32,[None,784])
    W=tf.Variable(tf.zeros([784,10]))
    b=tf.Variable(tf.zeros([10]))
    y=tf.nn.softmax(tf.matmul(x,W)+b)
    #损失函数，使用交叉熵
    y_=tf.placeholder(tf.float32,[None,10])
    cross_entropy=tf.reduce_mean(-tf.reduce_sum(y_*tf.log(y),reduction_indices=[1]))
    #优化算法：随机梯度下降SGD，最小化交叉熵，得到训练操作，学习率为0.5
    train_step=tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
    #使用全局参数初始化器，执行run
    sess.run(tf.global_variables_initializer())
    #逐批训练数据
    for i in range(1000):
        batch_xs,batch_ys=mnist.train.next_batch(100)
        sess.run(train_step,feed_dict={x:batch_xs,y_:batch_ys})
    #验证准确率
    correct_prediction=tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
    accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
    result=sess.run(accuracy,feed_dict={x:mnist.test.images,y_:mnist.test.labels})
    print(result)


