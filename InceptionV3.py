'''
2018-09-08
Inception V3网络
42层
'''
import math
import tensorflow as tf
slim=tf.contrib.slim
trunc_normal=lambda stddev:tf.truncated_normal_initializer(0.,stddev)
import time
from datetime import datetime

def inception_v3_arg_scope(weight_decay=0.00004,stddev=0.1,
                           batch_norm_var_collection='moving_vars'):
    '''
    生成网络中常用的函数的默认参数
    :param weight_decay: L2正则的权重衰减率
    :param stddev: 标准差
    :param batch_norm_var_collection: 参数
    :return:
    '''
    #参数字典
    batch_norm_params={
        'decay':0.997,  #衰减系数
        'epsilon':0.001,
        'updates_collections':tf.GraphKeys.UPDATE_OPS,
        'variables_collections':{
            'beta':None,
            'gamma':None,
            'moving_mean':[batch_norm_var_collection],
            'moving_variance':[batch_norm_var_collection]
        }
    }
    #slim.arg_scope可以给函数参数自动赋予某些默认值
    with slim.arg_scope([slim.conv2d,slim.fully_connected],
                        weights_regularizer=slim.l2_regularizer(weight_decay)):
        #定义一个卷积层的参数
        with slim.arg_scope(
            [slim.conv2d],
            weights_initializer=tf.truncated_normal_initializer(stddev=stddev),
            activation_fn=tf.nn.relu,
            normalizer_fn=slim.batch_norm,  #标准化器
            normalizer_params=batch_norm_params) as sc: #标准化器参数
            return sc
def inception_v3_base(inputs,scope=None):
    #生成Inception V3的卷积部分

    end_points={}   #用来保存关键节点供之后使用
    with tf.variable_scope(scope,'InceptionV3',[inputs]):
        with slim.arg_scope([slim.conv2d,slim.max_pool2d,slim.avg_pool2d],
                            stride=1,padding='VALID'):
            #非IM的卷积层，一行代码定义一层
            #输入299*299*3，最后输出35*35*192
            net=slim.conv2d(inputs,32,[3,3],stride=2,scope='Conv2d_1a_3x3')
            net=slim.conv2d(net,32,[3,3],scope='Conv2d_2a_3x3')
            net = slim.conv2d(net, 64, [3, 3], padding='SAME',scope='Conv2d_2b_3x3')
            net = slim.max_pool2d(net, [3, 3], stride=2, scope='MaxPool_3a_3x3')
            net = slim.conv2d(net, 80, [1, 1], scope='Conv2d_3b_3x3')
            net = slim.conv2d(net, 192, [3, 3], scope='Conv2d_4a_3x3')
            net = slim.max_pool2d(net,  [3, 3], stride=2,scope='MaxPool_5a_3x3')

    #接下来是三个连续的Inception模块组
    #第一个Inception组
    with slim.arg_scope([slim.conv2d,slim.max_pool2d,slim.avg_pool2d],
                        stride=1,padding='SAME'):
        #第一个IM，4个分支分别输出，再用tf.concat组装
        with tf.variable_scope('Mixed_5b'):
            with tf.variable_scope('Branch_0'):
                branch_0=slim.conv2d(net,64,[1,1],scope='Conv2d_0a_1x1')
            with tf.variable_scope('Branch_1'):
                branch_1=slim.conv2d(net,48,[1,1],scope='Conv2d_0a_1x1')
                branch_1 = slim.conv2d(branch_1, 64, [5, 5], scope='Conv2d_0b_5x5')
            with tf.variable_scope('Branch_2'):
                branch_2=slim.conv2d(net,64,[1,1],scope='Conv2d_0a_1x1')
                branch_2 = slim.conv2d(branch_2, 96, [3, 3], scope='Conv2d_0b_3x3')
                branch_2 = slim.conv2d(branch_2, 96, [3, 3], scope='Conv2d_0c_3x3')
            with tf.variable_scope('Branch_3'):
                branch_3 = slim.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
                branch_3 = slim.conv2d(branch_3, 32, [1, 1], scope='Conv2d_0b_1x1')
            #合并，3表示在第三个维度合并，输出35*35*256
            net=tf.concat([branch_0,branch_1,branch_2,branch_3],3)
        #第二个IM，与第一个相比最后通道数变成64，输出35*35*288
        with tf.variable_scope('Mixed_5c'):
            with tf.variable_scope('Branch_0'):
                branch_0 = slim.conv2d(net, 64, [1, 1], scope='Conv2d_0a_1x1')
            with tf.variable_scope('Branch_1'):
                branch_1 = slim.conv2d(net, 48, [1, 1], scope='Conv2d_0b_1x1')
                branch_1 = slim.conv2d(branch_1, 64, [5, 5], scope='Conv2d_0c_5x5')
            with tf.variable_scope('Branch_2'):
                branch_2 = slim.conv2d(net, 64, [1, 1], scope='Conv2d_1_0a_1x1')
                branch_2 = slim.conv2d(branch_2, 96, [3, 3], scope='Conv2d_0b_3x3')
                branch_2 = slim.conv2d(branch_2, 96, [3, 3], scope='Conv2d_0c_3x3')
            with tf.variable_scope('Branch_3'):
                branch_3 = slim.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
                branch_3 = slim.conv2d(branch_3, 64, [1, 1], scope='Conv2d_0b_1x1')
            net = tf.concat([branch_0, branch_1, branch_2, branch_3], 3)
        #第三个IM，与第二个相同，输出35*35*288
        with tf.variable_scope('Mixed_5d'):
            with tf.variable_scope('Branch_0'):
                branch_0 = slim.conv2d(net, 64, [1, 1], scope='Conv2d_0a_1x1')
            with tf.variable_scope('Branch_1'):
                branch_1 = slim.conv2d(net, 48, [1, 1], scope='Conv2d_0b_1x1')
                branch_1 = slim.conv2d(branch_1, 64, [5, 5], scope='Conv2d_0c_5x5')
            with tf.variable_scope('Branch_2'):
                branch_2 = slim.conv2d(net, 64, [1, 1], scope='Conv2d_1_0a_1x1')
                branch_2 = slim.conv2d(branch_2, 96, [3, 3], scope='Conv2d_0b_3x3')
                branch_2 = slim.conv2d(branch_2, 96, [3, 3], scope='Conv2d_0c_3x3')
            with tf.variable_scope('Branch_3'):
                branch_3 = slim.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
                branch_3 = slim.conv2d(branch_3, 64, [1, 1], scope='Conv2d_0b_1x1')
            net = tf.concat([branch_0, branch_1, branch_2, branch_3], 3)

        #第二个Inception组，均输出17*17*768
        #第一个IM，包含3个分支
        with tf.variable_scope('Mixed_6a'):
            with tf.variable_scope('Branch_0'):
                branch_0 = slim.conv2d(net, 384, [3, 3], stride=2,padding='VALID',
                                       scope='Conv2d_1a_1x1')
            with tf.variable_scope('Branch_1'):
                branch_1 = slim.conv2d(net, 64, [1, 1], scope='Conv2d_0a_1x1')
                branch_1 = slim.conv2d(branch_1, 96, [3, 3], scope='Conv2d_0b_3x3')
                branch_1 = slim.conv2d(branch_1, 96, [3, 3],stride=2,
                                       padding='VALID',scope='Conv2d_1a_1x1')
            with tf.variable_scope('Branch_2'):
                branch_2 = slim.max_pool2d(net, [3, 3], stride=2,padding='VALID',
                                           scope='MaxPool_1a_3x3')
            net = tf.concat([branch_0, branch_1, branch_2], 3)
        #第二个IM，包含4个分支，输出17*17*768
        with tf.variable_scope('Mixed_6b'):
            with tf.variable_scope('Branch_0'):
                branch_0 = slim.conv2d(net, 192, [3, 3], scope='Conv2d_0a_1x1')
            with tf.variable_scope('Branch_1'):
                branch_1 = slim.conv2d(net, 128, [1, 1], scope='Conv2d_0a_1x1')
                branch_1 = slim.conv2d(branch_1, 128, [1, 7], scope='Conv2d_0b_1x7')
                branch_1 = slim.conv2d(branch_1, 192, [7, 1], scope='Conv2d_0c_7x1')
            with tf.variable_scope('Branch_2'):
                branch_2 = slim.conv2d(net,128, [1, 1], scope='Conv2d_0a_1x1')
                branch_2 = slim.conv2d(branch_2, 128, [7, 1], scope='Conv2d_0b_7x1')
                branch_2 = slim.conv2d(branch_2, 128, [1, 7], scope='Conv2d_0c_1x7')
                branch_2 = slim.conv2d(branch_2, 128, [7, 1], scope='Conv2d_0d_7x1')
                branch_2 = slim.conv2d(branch_2, 192, [1, 7], scope='Conv2d_0e_1x7')
            with tf.variable_scope('Branch_3'):
                branch_3 = slim.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
                branch_3 = slim.conv2d(branch_3, 192, [1, 1], scope='Conv2d_0b_1x1')
            net = tf.concat([branch_0, branch_1, branch_2,branch_3], 3)
        #第三个IM，与第二个大致相同，除了将128变为160
        with tf.variable_scope('Mixed_6c'):
            with tf.variable_scope('Branch_0'):
                branch_0 = slim.conv2d(net, 192, [3, 3], scope='Conv2d_0a_1x1')
            with tf.variable_scope('Branch_1'):
                branch_1 = slim.conv2d(net, 160, [1, 1], scope='Conv2d_0a_1x1')
                branch_1 = slim.conv2d(branch_1, 160, [1, 7], scope='Conv2d_0b_1x7')
                branch_1 = slim.conv2d(branch_1, 192, [7, 1], scope='Conv2d_0c_7x1')
            with tf.variable_scope('Branch_2'):
                branch_2 = slim.conv2d(net,160, [1, 1], scope='Conv2d_0a_1x1')
                branch_2 = slim.conv2d(branch_2, 160, [7, 1], scope='Conv2d_0b_7x1')
                branch_2 = slim.conv2d(branch_2, 160, [1, 7], scope='Conv2d_0c_1x7')
                branch_2 = slim.conv2d(branch_2, 160, [7, 1], scope='Conv2d_0d_7x1')
                branch_2 = slim.conv2d(branch_2, 192, [1, 7], scope='Conv2d_0e_1x7')
            with tf.variable_scope('Branch_3'):
                branch_3 = slim.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
                branch_3 = slim.conv2d(branch_3, 192, [1, 1], scope='Conv2d_0b_1x1')
            net = tf.concat([branch_0, branch_1, branch_2,branch_3], 3)
        #第四个IM，与第三个相同
        with tf.variable_scope('Mixed_6d'):
            with tf.variable_scope('Branch_0'):
                branch_0 = slim.conv2d(net, 192, [3, 3], scope='Conv2d_0a_1x1')
            with tf.variable_scope('Branch_1'):
                branch_1 = slim.conv2d(net, 160, [1, 1], scope='Conv2d_0a_1x1')
                branch_1 = slim.conv2d(branch_1, 160, [1, 7], scope='Conv2d_0b_1x7')
                branch_1 = slim.conv2d(branch_1, 192, [7, 1], scope='Conv2d_0c_7x1')
            with tf.variable_scope('Branch_2'):
                branch_2 = slim.conv2d(net,160, [1, 1], scope='Conv2d_0a_1x1')
                branch_2 = slim.conv2d(branch_2, 160, [7, 1], scope='Conv2d_0b_7x1')
                branch_2 = slim.conv2d(branch_2, 160, [1, 7], scope='Conv2d_0c_1x7')
                branch_2 = slim.conv2d(branch_2, 160, [7, 1], scope='Conv2d_0d_7x1')
                branch_2 = slim.conv2d(branch_2, 192, [1, 7], scope='Conv2d_0e_1x7')
            with tf.variable_scope('Branch_3'):
                branch_3 = slim.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
                branch_3 = slim.conv2d(branch_3, 192, [1, 1], scope='Conv2d_0b_1x1')
            net = tf.concat([branch_0, branch_1, branch_2,branch_3], 3)
        #第五个IM，与前两个相同
        with tf.variable_scope('Mixed_6e'):
            with tf.variable_scope('Branch_0'):
                branch_0 = slim.conv2d(net, 192, [3, 3], scope='Conv2d_0a_1x1')
            with tf.variable_scope('Branch_1'):
                branch_1 = slim.conv2d(net, 160, [1, 1], scope='Conv2d_0a_1x1')
                branch_1 = slim.conv2d(branch_1, 160, [1, 7], scope='Conv2d_0b_1x7')
                branch_1 = slim.conv2d(branch_1, 192, [7, 1], scope='Conv2d_0c_7x1')
            with tf.variable_scope('Branch_2'):
                branch_2 = slim.conv2d(net,160, [1, 1], scope='Conv2d_0a_1x1')
                branch_2 = slim.conv2d(branch_2, 160, [7, 1], scope='Conv2d_0b_7x1')
                branch_2 = slim.conv2d(branch_2, 160, [1, 7], scope='Conv2d_0c_1x7')
                branch_2 = slim.conv2d(branch_2, 160, [7, 1], scope='Conv2d_0d_7x1')
                branch_2 = slim.conv2d(branch_2, 192, [1, 7], scope='Conv2d_0e_1x7')
            with tf.variable_scope('Branch_3'):
                branch_3 = slim.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
                branch_3 = slim.conv2d(branch_3, 192, [1, 1], scope='Conv2d_0b_1x1')
            net = tf.concat([branch_0, branch_1, branch_2,branch_3], 3)
        #将第二个Inception组的输出存储起来，作为分类辅助
        end_points['Mixed_6e']=net

        #第三个Inception组
        #第一个IM，包含3个分支，输出8*8*1280
        with tf.variable_scope('Mixed_7a'):
            with tf.variable_scope('Branch_0'):
                branch_0 = slim.conv2d(net, 192, [1, 1], scope='Conv2d_0a_1x1')
                branch_0 = slim.conv2d(branch_0, 320, [3, 3], stride=2,
                                       padding='VALID',scope='Conv2d_1a_3x3')
            with tf.variable_scope('Branch_1'):
                branch_1 = slim.conv2d(net, 192, [1, 1], scope='Conv2d_0a_1x1')
                branch_1 = slim.conv2d(branch_1, 192, [1, 7], scope='Conv2d_0b_1x7')
                branch_1 = slim.conv2d(branch_1, 192, [7, 1], scope='Conv2d_0c_7x1')
                branch_1 = slim.conv2d(branch_1, 192, [3, 3], stride=2,
                                       padding='VALID', scope='Conv2d_1a_3x3')
            with tf.variable_scope('Branch_2'):
                branch_2 = slim.max_pool2d(net, [3, 3], stride=2,padding='VALID',
                                           scope='MaxPool_1a_3x3')
            net = tf.concat([branch_0, branch_1, branch_2], 3)
        # 第二个IM，输出8*8*2048
        with tf.variable_scope('Mixed_7b'):
            with tf.variable_scope('Branch_0'):
                branch_0 = slim.conv2d(net, 320, [1, 1], scope='Conv2d_0a_1x1')
            with tf.variable_scope('Branch_1'):
                branch_1 = slim.conv2d(net, 384, [1, 1], scope='Conv2d_0a_1x1')
                branch_1 = tf.concat([
                    slim.conv2d(branch_1, 384, [1, 3], scope='Conv2d_0b_1x3'),
                    slim.conv2d(branch_1, 384, [3, 1], scope='Conv2d_0b_3x1')],3)
            with tf.variable_scope('Branch_2'):
                branch_2 = slim.conv2d(net, 448, [1, 1], scope='Conv2d_0a_1x1')
                branch_2 = slim.conv2d(branch_2, 384, [3, 3], scope='Conv2d_0b_3x3')
                branch_2 = tf.concat([
                     slim.conv2d(branch_2, 384, [1, 3], scope='Conv2d_0c_1x3'),
                     slim.conv2d(branch_2, 384, [3, 1], scope='Conv2d_0d_3x1')],3)
            with tf.variable_scope('Branch_3'):
                branch_3 = slim.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
                branch_3 = slim.conv2d(branch_3, 192, [1, 1], scope='Conv2d_0b_1x1')
            net = tf.concat([branch_0, branch_1, branch_2, branch_3], 3)
        # 第三个IM，和第二个一样，输出8*8*2048
        with tf.variable_scope('Mixed_7c'):
            with tf.variable_scope('Branch_0'):
                branch_0 = slim.conv2d(net, 320, [1, 1], scope='Conv2d_0a_1x1')
            with tf.variable_scope('Branch_1'):
                branch_1 = slim.conv2d(net, 384, [1, 1], scope='Conv2d_0a_1x1')
                branch_1 = tf.concat([
                    slim.conv2d(branch_1, 384, [1, 3], scope='Conv2d_0b_1x3'),
                    slim.conv2d(branch_1, 384, [3, 1], scope='Conv2d_0b_3x1')],3)
            with tf.variable_scope('Branch_2'):
                branch_2 = slim.conv2d(net, 448, [1, 1], scope='Conv2d_0a_1x1')
                branch_2 = slim.conv2d(branch_2, 384, [3, 3], scope='Conv2d_0b_3x3')
                branch_2 = tf.concat([
                     slim.conv2d(branch_2, 384, [1, 3], scope='Conv2d_0c_1x3'),
                     slim.conv2d(branch_2, 384, [3, 1], scope='Conv2d_0d_3x1')],3)
            with tf.variable_scope('Branch_3'):
                branch_3 = slim.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
                branch_3 = slim.conv2d(branch_3, 192, [1, 1], scope='Conv2d_0b_1x1')
            net = tf.concat([branch_0, branch_1, branch_2, branch_3], 3)
        #返回这个Inception的结果
        return net,end_points


def inception_v3(inputs,num_classes=1000,is_training=True,dropout_keep_prob=0.8,
                 prediciton_fn=slim.softmax,spatial_squeeze=True,reuse=None,
                 scope="InceptionV3"):
    '''
    实现IV3的全局平均池化、softmax和auxiliary logits
    :param inputs: 输入
    :param num_classes: 最后需要分类的数量
    :param is_training: 是否是训练过程，只有训练时才气用BN和DO
    :param dropout_keep_prob: 训练时保留节点的比例
    :param prediciton_fn: 最后用来分类的函数
    :param spatial_squeeze: 是否对输出进行squeeze（去除维数为1的维度）
    :param reuse: 是否对网络和变量进行重复使用
    :param scope: 函数默认参数的环境
    :return:
    '''
    #定义参数默认值
    with tf.variable_scope(scope,'InceptionV3',
                           [inputs,num_classes],reuse=reuse) as scope:
        #构建卷积部分
        with slim.arg_scope([slim.batch_norm,slim.dropout],is_training=is_training):
            net,end_points=inception_v3_base(inputs,scope=scope)
            #处理AL，AL是辅助分类的节点
            with slim.arg_scope([slim.conv2d,slim.max_pool2d,slim.avg_pool2d],
                                stride=1,padding='SAME'):
                aux_logits=end_points['Mixed_6e']
                with tf.variable_scope('AuxLogits'):
                    aux_logits=slim.avg_pool2d(
                        aux_logits,[5,5],stride=3,padding='VALID',scope='AvgPool_1a_5x5')
                    aux_logits=slim.conv2d(aux_logits,128,[1,1],scope='Conv2d_1b_1x1')
                    aux_logits=slim.conv2d(
                        aux_logits,768,[5,5],
                        weights_initializer=trunc_normal(0.01),
                        padding='VALID',scope='Conv2d_2a_5x5')
                    aux_logits=slim.conv2d(
                        aux_logits,num_classes,[1,1],activation_fn=None,normalizer_fn=None,
                        weights_initializer=trunc_normal(0.001),scope='Conv2d_2b_1x1')
                    #将输出1*1*1000,并消除前两个维度为1的维度
                    if spatial_squeeze:
                        aux_logits=tf.squeeze(aux_logits,[1,2],name='SaptialSqueeze')
                    end_points['AuxLogits']=aux_logits
            #正常预测分类的逻辑
            with tf.variable_scope('Logits'):
                net=slim.avg_pool2d(net,[8,8],padding='VALID',scope='AvgPool_1a_8x8')
                net=slim.dropout(net,keep_prob=dropout_keep_prob,scope='Dropout_1b')
                end_points['PreLogits']=net
                logits=slim.conv2d(net,num_classes,[1,1],activation_fn=None,
                                   normalizer_fn=None,scope='Conv2d_1c_1x1')
                if spatial_squeeze:
                    logits=tf.squeeze(logits,[1,2],name='SpatialSqueeze')
            end_points['Logits']=logits
            end_points['Predictions']=prediciton_fn(logits,scope='Predictions')
    return logits,end_points

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
    mn=total_duration/num_batches   #每批数据所耗费的时间
    vr=total_duration_squared/num_batches-mn*mn
    sd=math.sqrt(vr)
    print('%s : %s across %d steps, %.3f +/- %.3f sec/batch' %
              (datetime.now(),info_string,num_batches,mn,sd))

batch_size=32
height,width=299,299
inputs=tf.random_uniform((batch_size,height,width,3))
with slim.arg_scope(inception_v3_arg_scope()):
    logits,end_points=inception_v3(inputs,is_training=False)

init=tf.global_variables_initializer()
sess=tf.Session()
sess.run(init)
num_batches=100
time_tensorflow_run(sess,logits,'Forward')

