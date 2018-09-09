'''
2018-09-08
ResNet V2
使用contrib.slim库来辅助构建
'''

import collections
import tensorflow as tf
import time
from datetime import datetime
import math
slim=tf.contrib.slim

#使用collections.namedtuple设计ResNet基本Block模块组，来创建Block的类
#只包含数据结构，不包含具体方法
class Block(collections.namedtuple('Block',['scope','unit_fn','args'])):
    #参数为名称、残差学习单元和元素列表
    'A named tuple describing a ResNet block.'

def subsample(inputs,factor,scope=None):
    #降采样
    if factor==1:
        return inputs
    else:
        return slim.max_pool2d(inputs,[1,1],stride=factor,scope=scope)

def conv2d_same(inputs,num_outputs,kernel_size,stride,scope=None):
    #创建卷积层
    if stride==1:
        return slim.conv2d(inputs,num_outputs,kernel_size,stride=1,
                           padding='SAME',scope=scope)
    else:
        pad_total=kernel_size-1
        pad_beg=pad_total//2
        pad_end=pad_total-pad_beg
        inputs=tf.pad(inputs,[[0,0],[pad_beg,pad_end],
                              [pad_beg,pad_end],[0,0]])
        return slim.conv2d(inputs,num_outputs,kernel_size,stride=stride,
                           padding='VALID',scope=scope)

@slim.add_arg_scope
def stack_blocks_dense(net,blocks,outputs_collections=None):
    '''
    接下来定义堆叠Blocks的函数
    :param net: 输入
    :param blocks: 之前定义的Block的class的列表
    :param outputs_collections:用来搜集各个end_points
    :return:
    '''
    #两层循环，逐个Block，逐个residual unit堆叠
    for block in blocks:
        with tf.variable_scope(block.scope,'block',[net]) as sc:
            for i, unit in enumerate(block.args):
                with tf.variable_scope('unit_%d' % (i+1),values=[net]):
                    #拿到每个Block中每个residual units的args，展开
                    unit_depth,unit_depth_bottleneck,unit_stride=unit
                    #创建残差学习单元
                    net=block.unit_fn(net,depth=unit_depth, #第三层输出通道数
                                      depth_bottleneck=unit_depth_bottleneck,   #前两层输出通道数
                                      stride=unit_stride)   #中间那层的步长
                #将输出net添加到collection中
                net=slim.utils.collect_named_outputs(outputs_collections,sc.name,net)
    return net

def resnet_arg_scope(is_training=True,weight_decay=0.0001,batch_norm_decay=0.997,
                     batch_norm_epsilon=1e-5,batch_norm_scale=True):
    #定义某些函数的参数默认值
    batch_norm_params={
        'is_training':is_training,
        'decay':batch_norm_decay,
        'epsilon':batch_norm_epsilon,
        'scale':batch_norm_scale,
        'update_collections':tf.GraphKeys.UPDATE_OPS,
    }

    with slim.arg_scope([slim.conv2d],
        weights_regularizer=slim.l2_regularizer(weight_decay),  #设置L2正则器
        weights_initializer=slim.variance_scaling_initializer(),#设置权重初始化器
        activation_fn=tf.nn.relu,   #设置激活函数
        normalizer_fn=slim.batch_norm,  #设置标准化器
        normalizer_params=batch_norm_params):
        with slim.arg_scope([slim.batch_norm],**batch_norm_params):
            with slim.arg_scope([slim.max_pool2d],padding='SAME') as arg_sc:  #设置最大池化的padding模式
                return arg_sc

@slim.add_arg_scope
def bottleneck(inputs,depth,depth_bottleneck,stride,
               outputs_collections=None,scope=None):
    #定义bottleneck残差学习单元，它每一层前都用了BN，
    # 对输入进行preactivation而不是在卷积进行激活函数处理
    with tf.variable_scope(scope,'bottleneck_v2',[inputs]) as sc:
        #获取输入的最后一个维度，即输出通道数，限定为最少4个维度
        depth_in=slim.utils.last_dimension(inputs.get_shape(),min_rank=4)
        #对输入进行BN，并进行预激活
        preact=slim.batch_norm(inputs,activation_fn=tf.nn.relu,scope='preact')
        if depth == depth_in:   #降采样使通道数与残差一致
            shortcut=subsample(inputs,stride,'shortcut')
        else:   #使输出与输入一致
            shortcut=slim.conv2d(preact,depth,[1,1],stride=stride,normalizer_fn=None,
                                 activation_fn=None,scope='shortcut')
        #定义残差residual，有三层
        residual=slim.conv2d(preact,depth_bottleneck,3,stride=1,scope='conv1')
        residual=conv2d_same(residual,depth_bottleneck,3,stride,scope='conv2')
        residual=slim.conv2d(residual,depth,[1,1],stride=1,normalizer_fn=None,
                             activation_fn=None,scope='conv3')
        output=shortcut+residual
        #得到最后output并添加到collection中
        return slim.utils.collect_named_outputs(outputs_collections,sc.name,output)

def resnet_v2(inputs,blocks,num_classes=None,global_pool=True,
              include_root_block=True,reuse=None,scope=None):
    #include_root_block表示是否加上ResNet网络最前面常用的7*7卷积和最大池化
    #global_pool表示是否加上最后一层全局平均池化
    #先设置参数
    with tf.variable_scope(scope,'resnet_v2',[inputs],reuse=reuse) as sc:
        end_points_collection=sc.original_name_scope+'_end_points'
        with slim.arg_scope([slim.conv2d,bottleneck,stack_blocks_dense],
                            outputs_collections=end_points_collection):
            net=inputs
            if include_root_block:
                with slim.arg_scope([slim.conv2d],activation_fn=None,normalizer_fn=None):
                    net=conv2d_same(net,64,7,stride=2,scope='conv1')
                net=slim.max_pool2d(net,[3,3],stride=2,scope='pool1')
            net=stack_blocks_dense(net,blocks)  #生成残差学习模块组
            net=slim.batch_norm(net,activation_fn=tf.nn.relu,scope='postnorm')
            if global_pool: #根据标记添加全局平均池化层
                net=tf.reduce_mean(net,[1,2],name='pool5',keep_dims=True)
            if num_classes is not None: #根据标记输出通道
                net=slim.conv2d(net,num_classes,[1,1],activation_fn=None,
                                normalizer_fn=None,scope='logits')
                #将collection转化为dict
                end_points=slim.utils.convert_collection_to_dict(end_points_collection)
                if num_classes is not None: #输出网络结果
                    end_points['predictions']=slim.softmax(net,scope='predictions')
                return net,end_points

def resnet_v2_50(inputs,num_classes=None,global_pool=True,
                 reuse=None,scope='resnet_v2_50'):
    #残差学习单元分别为3，4，6，3，共16*3+2=50层
    blocks=[
        Block('block1',bottleneck,[(256,64,1)]*2+[(256,64,2)]),
        Block('block2', bottleneck, [(512, 128, 1)] * 3 + [(512, 128, 2)]),
        Block('block1', bottleneck, [(1024, 256, 1)] * 5 + [(1024, 256, 2)]),
        Block('block1', bottleneck, [(2048, 512, 1)] * 3),
    ]
    return resnet_v2(inputs,blocks,num_classes,global_pool,include_root_block=True,reuse=reuse,scope=scope)

def resnet_v2_101(inputs,num_classes=None,global_pool=True,
                 reuse=None,scope='resnet_v2_101'):
    #残差学习单元分别为3，4，23，3，共33*3+2=101层
    blocks=[
        Block('block1',bottleneck,[(256,64,1)]*2+[(256,64,2)]),
        Block('block2', bottleneck, [(512, 128, 1)] * 3 + [(512, 128, 2)]),
        Block('block1', bottleneck, [(1024, 256, 1)] * 22 + [(1024, 256, 2)]),
        Block('block1', bottleneck, [(2048, 512, 1)] * 3),
    ]
    return resnet_v2(inputs,blocks,num_classes,global_pool,include_root_block=True,reuse=reuse,scope=scope)

def resnet_v2_152(inputs,num_classes=None,global_pool=True,
                 reuse=None,scope='resnet_v2_152'):
    #残差学习单元分别为3，8，36，3，共50*3+2=152层
    blocks=[
        Block('block1',bottleneck,[(256,64,1)]*2+[(256,64,2)]),
        Block('block2', bottleneck, [(512, 128, 1)] * 7 + [(512, 128, 2)]),
        Block('block1', bottleneck, [(1024, 256, 1)] * 35 + [(1024, 256, 2)]),
        Block('block1', bottleneck, [(2048, 512, 1)] * 3),
    ]
    return resnet_v2(inputs,blocks,num_classes,global_pool,include_root_block=True,reuse=reuse,scope=scope)

def resnet_v2_200(inputs,num_classes=None,global_pool=True,
                 reuse=None,scope='resnet_v2_200'):
    #残差学习单元分别为3，24，36，3，共66*3+2=200层
    blocks=[
        Block('block1',bottleneck,[(256,64,1)]*2+[(256,64,2)]),
        Block('block2', bottleneck, [(512, 128, 1)] * 23 + [(512, 128, 2)]),
        Block('block1', bottleneck, [(1024, 256, 1)] * 35 + [(1024, 256, 2)]),
        Block('block1', bottleneck, [(2048, 512, 1)] * 3),
    ]
    return resnet_v2(inputs, blocks, num_classes, global_pool, include_root_block=True, reuse=reuse, scope=scope)

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

batch_size = 32
height, width = 224, 224
inputs = tf.random_uniform((batch_size, height, width, 3))
with slim.arg_scope(resnet_arg_scope(is_training=False)):
    net, end_points = resnet_v2_152(inputs, 1000)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
num_batches = 100
time_tensorflow_run(sess, net, 'Forward')
