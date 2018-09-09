'''
2018-09-05
AlexNet
CNN的一种
'''

from datetime import datetime
import math
import time
import tensorflow as tf

batch_size=32
num_batches=100

def print_activations(t):
    #接受一个tensor并输出名称和尺寸
    print(t.op.name,' ',t.get_shape().as_list())

def inference(images):
    #接受images作为输入，返回最后一层pool5与parameters
    parameters=[]
    #第一层卷积层，64卷积核
    #将scope内生成的Variable自动命名为conv1/xxx
    with tf.name_scope('conv1') as scope:
        kernel=tf.Variable(tf.truncated_normal([11,11,3,64],
                           dtype=tf.float32,stddev=1e-1),name='weights')
        conv=tf.nn.conv2d(images,kernel,[1,4,4,1],padding='SAME')
        biases=tf.Variable(tf.constant(0.0,shape=[64],dtype=tf.float32),
                           trainable=True,name='biases')
        bias=tf.nn.bias_add(conv,biases)
        conv1=tf.nn.relu(bias,name=scope)
        print_activations(conv1)
        parameters+=[kernel,biases]

    #输入为conv1，depth_radius设为4，bias设为1，alpha，beta都是推荐值
    lrn1=tf.nn.lrn(conv1,4,bias=1.0,alpha=0.001/9,beta=0.75,name='lrn1')
    #VALID表示取样时不能超过边框，SAME是可以填充边界外的值
    pool1=tf.nn.max_pool(lrn1,ksize=[1,3,3,1],strides=[1,2,2,1],
                         padding='VALID',name='pool1')
    print_activations(pool1)

    #第二个卷积层，192卷积核
    with tf.name_scope('conv2') as scope:
        kernel=tf.Variable(tf.truncated_normal([5,5,64,192],dtype=tf.float32,
                                               stddev=1e-1),name='weights')
        conv=tf.nn.conv2d(pool1,kernel,[1,1,1,1],padding='SAME')
        biases=tf.Variable(tf.constant(0.0,shape=[192],
                                       dtype=tf.float32),trainable=True,name='biases')
        bias=tf.nn.bias_add(conv,biases)
        conv2=tf.nn.relu(bias,name=scope)
        parameters+=[kernel,biases]
    print_activations(conv2)

    lrn2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001 / 9, beta=0.75, name='lrn2')
    pool2 = tf.nn.max_pool(lrn2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                           padding='VALID', name='pool2')
    print_activations(pool2)

    #第三个卷积层，384卷积核
    with tf.name_scope('conv3') as scope:
        kernel=tf.Variable(tf.truncated_normal([3,3,192,384],dtype=tf.float32,
                                               stddev=1e-1),name='weights')
        conv=tf.nn.conv2d(pool2,kernel,[1,1,1,1],padding='SAME')
        biases=tf.Variable(tf.constant(0.0,shape=[384],
                                       dtype=tf.float32),trainable=True,name='biases')
        bias=tf.nn.bias_add(conv,biases)
        conv3=tf.nn.relu(bias,name=scope)
        parameters+=[kernel,biases]
    print_activations(conv3)

    #第四个卷积层，256卷积核
    with tf.name_scope('conv4') as scope:
        kernel=tf.Variable(tf.truncated_normal([3,3,384,256],dtype=tf.float32,
                                               stddev=1e-1),name='weights')
        conv=tf.nn.conv2d(conv3,kernel,[1,1,1,1],padding='SAME')
        biases=tf.Variable(tf.constant(0.0,shape=[256],
                                       dtype=tf.float32),trainable=True,name='biases')
        bias=tf.nn.bias_add(conv,biases)
        conv4=tf.nn.relu(bias,name=scope)
        parameters+=[kernel,biases]
    print_activations(conv4)

    #第五个卷积层，256卷积核
    with tf.name_scope('conv5') as scope:
        kernel=tf.Variable(tf.truncated_normal([3,3,256,256],dtype=tf.float32,
                                               stddev=1e-1),name='weights')
        conv=tf.nn.conv2d(conv4,kernel,[1,1,1,1],padding='SAME')
        biases=tf.Variable(tf.constant(0.0,shape=[256],
                                       dtype=tf.float32),trainable=True,name='biases')
        bias=tf.nn.bias_add(conv,biases)
        conv5=tf.nn.relu(bias,name=scope)
        parameters+=[kernel,biases]
    print_activations(conv5)

    #最大池化层
    pool5 = tf.nn.max_pool(conv5, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                           padding='VALID', name='pool5')
    print_activations(pool5)
    return pool5,parameters

    #在正式使用AlexNet的时候还需要自行添加3个节点数分别为4096，4096，1000的全连接层

def time_tensorflow_run(session,target,info_string):
    '''
    计算每轮时间
    :param session: TensorFlow的session
    :param target: 需要评测的运算算子
    :param info_string: 测试的名称
    :return:
    '''
    #预热轮数，只考虑10轮之后的计算时间
    num_steps_burn_in=10
    total_duration=0.0
    total_duration_squared=0.0
    #迭代计算，10轮过后再打印时间
    for i in range(num_batches+num_steps_burn_in):
        start_time=time.time()
        _=session.run(target)   #执行迭代
        duration=time.time()-start_time #运行时间
        if i >= num_steps_burn_in:
            if not i%10:
                print('%s:step %d, duration=%.3f' % (datetime.now(),
                                             i-num_steps_burn_in,duration))
            total_duration+=duration
            total_duration_squared+=duration*duration

    mn=total_duration/num_batches   #平均耗时
    vr=total_duration_squared/num_batches-mn*mn #方差
    sd=math.sqrt(vr)    #标准差
    print('%s: %s across %d steps, %.3f +/- %.3f sec/batch' %
          (datetime.now(),info_string,num_batches,mn,sd))

def run_benchmark():
    #主函数
    #定义默认的Graph
    with tf.Graph().as_default():
        image_size=224
        #构造正态分布的随机tensor
        images=tf.Variable(tf.random_normal([batch_size,
                 image_size,image_size,3],dtype=tf.float32,stddev=1e-1))
        #构造整个AlexNet网络，得到最后一个pool5和参数集合
        pool5,parameters=inference(images)
        #初始化所有参赛并创建新的session
        init=tf.global_variables_initializer()
        sess=tf.Session()
        sess.run(init)
        #评测计算时间
        time_tensorflow_run(sess,pool5,'Forward')
        objective=tf.nn.l2_loss(pool5)
        #求所有模型参数的梯度，模拟训练过程
        grad=tf.gradients(objective,parameters)
        time_tensorflow_run(sess,grad,'Forward-backward')

#执行主函数
run_benchmark()








