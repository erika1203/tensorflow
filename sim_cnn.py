'''
2018-09-05
实现简单的卷积网络
tf04
'''

from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
mnist=input_data.read_data_sets('MNIST_data/',one_hot=True)
sess=tf.InteractiveSession()

def weight_variables(shape):
    #权重初始化，使用截断的正态分布噪声打破完全对称，标准差为0.1
    initial=tf.truncated_normal(shape,stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    #给偏置增加0.1噪声避免死亡节点
    initial=tf.constant(0.1,shape=shape)
    return tf.Variable(initial)

def conv2d(x,W):
    #2维卷积函数
    #strides表示卷积模板移动的步长，都是1表示不会遗漏每一个点
    #padding表示边界处理方式，SAME表示让卷积输入与输出保持同样尺寸
    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME')

def max_pool_2x2(x):
    #最大池化函数，最大池化会保留灰度值最高的像素
    #将2*2的像素降为1*1，strides的横竖两个方向以2为步长
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

#x为特征，y_为真是标签，输入转化为2D
#-1表示样本数量不固定，28*28是尺寸，1是颜色通道
x=tf.placeholder(tf.float32,[None,784])
y_=tf.placeholder(tf.float32,[None,10])
x_image=tf.reshape(x,[-1,28,28,1])

#定义第一个卷积层
#卷积层参数：卷积核尺寸5*5，1个颜色通道，32个卷积核
W_conv1=weight_variables([5,5,1,32])
b_conv1=bias_variable([32])
h_conv1=tf.nn.relu(conv2d(x_image,W_conv1)+b_conv1)
h_pool1=max_pool_2x2(h_conv1)

#定义第二个卷积层，提取64种特征
W_conv2=weight_variables([5,5,32,64])
b_conv2=bias_variable([64])
h_conv2=tf.nn.relu(conv2d(h_pool1,W_conv2)+b_conv2)
h_pool2=max_pool_2x2(h_conv2)

#经历上两个卷积操作，图片尺寸从28*28变为7*7
#第二个卷积核数量为64，输出的tensor为7*7*64
#将图片转为1D向量，连接一个全连接层，隐含节点1024，使用ReLU激活
W_fc1=weight_variables([7*7*64,1024])
b_fc1=bias_variable([1024])
h_pool2_flat=tf.reshape(h_pool2,[-1,7*7*64])
h_fc1=tf.nn.relu(tf.matmul(h_pool2_flat,W_fc1)+b_fc1)

#减轻过拟合，使用Dropout层
keep_prob=tf.placeholder(tf.float32)
h_fc1_drop=tf.nn.dropout(h_fc1,keep_prob)
#将Dropout的输出连接softmax层，得到最后的概率输出
W_fc2=weight_variables([1024,10])
b_fc2=bias_variable([10])
y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop,W_fc2)+b_fc2)
#定义损失函数和优化器
cross_entropy=tf.reduce_mean(-tf.reduce_sum(y_*tf.log(y_conv),reduction_indices=[1]))
train_step=tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
#准确率评测
correct_prediction=tf.equal(tf.argmax(y_conv,1),tf.argmax(y_,1))
accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

#开始训练过程，每100次训练会评测1次准确率以监控性能
tf.global_variables_initializer().run()
for i in range(20000):
    batch=mnist.train.next_batch(50)
    if i%100 ==0:
        train_accuracy=accuracy.eval(feed_dict={x:batch[0],y_:batch[1],
                                                keep_prob:1.0})
        print('step %d,training accuracy %.4f' % (i,train_accuracy))
    train_step.run(feed_dict={x:batch[0],y_:batch[1],keep_prob:0.5})

#测试
print('test accuracy %.4f' % accuracy.eval(feed_dict={x:mnist.test.images,y_:mnist.test.labels,keep_prob:1.0}))
