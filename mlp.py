'''
2018-09-05 多层感知机例子
只有一个隐含层，MNIST
tf03
'''

from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

def mlpMnist():
    mnist=input_data.read_data_sets('MNIST_data/',one_hot=True)
    sess=tf.InteractiveSession()
    in_units=784
    h1_units=300
    #将隐含层权重初始化为截断的正态分布，标准差为0.1
    #隐含层偏置和输出层参数设为0
    w1=tf.Variable(tf.truncated_normal([in_units,h1_units],stddev=0.1))
    b1=tf.Variable(tf.zeros([h1_units]))
    w2=tf.Variable(tf.zeros([h1_units,10]))
    b2=tf.Variable(tf.zeros([10]))
    x=tf.placeholder(tf.float32,[None,in_units])
    #Dropout的比率，即保留节点的概率
    keep_prob=tf.placeholder(tf.float32)
    #定义网络forward时的计算公式
    hidden1=tf.nn.relu(tf.matmul(x,w1)+b1)
    hidden1_drop=tf.nn.dropout(hidden1,keep_prob)
    y=tf.nn.softmax(tf.matmul(hidden1_drop,w2)+b2)
    y_=tf.placeholder(tf.float32,[None,10])
    #定义损失函数为交叉熵
    cross_entropy=tf.reduce_mean(-tf.reduce_sum(y_*tf.log(y),reduction_indices=[1]))
    #定义优化算法，使用自适应的Adagrad，学习速率设为0.3
    train_step=tf.train.AdagradOptimizer(0.3).minimize(cross_entropy)
    #训练步骤，一共30万样本，分为每批100个训练，实际上是5轮迭代
    #设置了0.75的keep_prob，也就是丢弃剩下的25%节点，将它们设为0
    tf.global_variables_initializer().run()
    for i in range(3000):
        batch_xs,batch_ys=mnist.train.next_batch(100)
        train_step.run({x:batch_xs,y_:batch_ys,keep_prob:0.75})
    #准确率评测，预测部分的keep_prob设为1
    correct_prediction=tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
    accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
    print(accuracy.eval({x:mnist.test.images,y_:mnist.test.labels,keep_prob:1.0}))



