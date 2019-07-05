import numpy as np
import tensorflow as tf
#-----------------------------------------------------------------------------------
# 给网络层创建权重张量
def get_variable(name, shape):
    return tf.get_variable(name=name,
                           shape=shape,
                           initializer=tf.initializers.random_normal(mean=0.0, stddev=0.02))
#-----------------------------------------------------------------------------------
# 卷积层
def conv(x, fmaps, kernel_size, stride=1, padding='SAME', use_bias=False, scope='conv'):
    channels = x.get_shape()[-1]

    with tf.variable_scope(scope):
        filters = get_variable(name='weights',
                               shape=[kernel_size, kernel_size, channels, fmaps])
        x = tf.nn.conv2d(input=x,
                         filter=filters,
                         strides=[1, stride, stride, 1],
                         padding=padding)

        if use_bias:
            bias = get_variable(name='bias', shape=[fmaps])
            x = tf.nn.bias_add(x, bias)

        return x
#-----------------------------------------------------------------------------------
# 逆卷积层
def deconv(x, fmaps, kernel_size, stride=2, padding='SAME', use_bias=False, scope='deconv'):
    channels = x.get_shape()[-1]
    height = int(x.get_shape()[1])# 含维度元素的list类
    width = int(x.get_shape()[2])

    with tf.variable_scope(scope):
        filters = get_variable(name='weights',
                               shape=[kernel_size, kernel_size, fmaps, channels])

        output_shape = [1, height*stride, width*stride, fmaps]
        x = tf.nn.conv2d_transpose(value=x,
                                   filter=filters,
                                   output_shape=output_shape,
                                   strides=[1, stride, stride, 1],
                                   padding=padding)

        if use_bias:
            bias = get_variable(name='bias', shape=[fmaps])
            x = tf.nn.bias_add(x, bias)

        return x
#-----------------------------------------------------------------------------------
# 激活函数
def relu(x):
    return tf.nn.relu(x)

def tanh(x):
    return tf.nn.tanh(x)

def leaky_relu(x, alpha):
    return tf.nn.leaky_relu(x, alpha)
#-----------------------------------------------------------------------------------
# batch norm
def batch_norm(x, scope='batch_norm'):
    return tf.contrib.layers.batch_norm(x,
                                        center=True,
                                        scale=True,
                                        epsilon=1e-5,
                                        scope=scope)
#-----------------------------------------------------------------------------------
# instance_norm
def instance_norm(x, scope='instance_norm'):
    return tf.contrib.layers.instance_norm(x,
                                           center=True,
                                           scale=True,
                                           epsilon=1e-5,
                                           scope=scope)
#-----------------------------------------------------------------------------------
# layer_norm
def layer_norm(x, scope='layer_norm') :
    return tf.contrib.layers.layer_norm(x,
                                        center=True,
                                        scale=True,
                                        scope=scope)
#-----------------------------------------------------------------------------------
# 残差块
def residual_block(x, fmaps, scope='resblock'):
    with tf.variable_scope(scope):
        f_x = conv(x, fmaps, kernel_size=3, stride=1, scope=scope+'_conv1')
        f_x = instance_norm(f_x, scope=scope+'_IN1')
        f_x = relu(f_x)

        f_x = conv(f_x, fmaps, kernel_size=3, stride=1, scope=scope+'_conv2')
        f_x = instance_norm(f_x, scope=scope+'_IN2')

        return relu(x+f_x)
#-----------------------------------------------------------------------------------
# 生成器
def generator(x, fmaps=64, reuse=False, scope='generator'):
    channels = x.get_shape()[-1]

    with tf.variable_scope(scope):
        if reuse:
            tf.get_variable_scope().reuse_variables()
        else:
            assert tf.get_variable_scope().reuse is False

        x = conv(x, fmaps, kernel_size=7, stride=1, scope='conv1')
        x = instance_norm(x, 'IN1')
        x = relu(x)

        x = conv(x, fmaps*2, kernel_size=3, stride=2, scope='conv2')
        x = conv(x, fmaps*2, kernel_size=3, stride=1, scope='conv2_')
        x = instance_norm(x, 'IN2')
        x = relu(x)

        x = conv(x, fmaps*4, kernel_size=3, stride=2, scope='conv3')
        x = conv(x, fmaps*4, kernel_size=3, stride=1, scope='conv3_')
        x = instance_norm(x, 'IN3')
        x = relu(x)

        for i in range(1, 9):
            x = residual_block(x, fmaps*4, scope='resblock'+str(i))

        x = deconv(x, fmaps*2, kernel_size=3, stride=2, scope='deconv1')
        x = conv(x, fmaps*2, kernel_size=3, stride=1, scope='conv4')
        x = instance_norm(x, 'IN4')

        x = deconv(x, fmaps, kernel_size=3, stride=2, scope='deconv2')
        x = conv(x, fmaps, kernel_size=3, stride=1, scope='conv5')
        x = instance_norm(x, 'IN5')

        x = conv(x, channels, kernel_size=7, stride=1, scope='conv_end')

        return tanh(x)
#-----------------------------------------------------------------------------------
# 判别器
def discriminator(x, fmaps=64, reuse=False, scope='discriminator'):
    with tf.variable_scope(scope, reuse=reuse):
        if reuse:
            tf.get_variable_scope().reuse_variables()
        else:
            assert tf.get_variable_scope().reuse is False

        x = conv(x, fmaps//2, kernel_size=3, stride=1, scope='conv1')
        x = leaky_relu(x, 0.2)
        
        x = conv(x, fmaps, kernel_size=3, stride=2, scope='conv2')
        x = leaky_relu(x, 0.2)
        x = conv(x, fmaps*2, kernel_size=3, stride=1, scope='conv3')
        x = instance_norm(x, 'IN1')
        x = leaky_relu(x, 0.2)

        x = conv(x, fmaps*2, kernel_size=3, stride=2, scope='conv4')
        x = leaky_relu(x, 0.2)
        x = conv(x, fmaps*4, kernel_size=3, stride=1, scope='conv5')
        x = instance_norm(x, 'IN2')
        x = leaky_relu(x, 0.2)
        
        x = conv(x, fmaps*4, kernel_size=3, stride=1, scope='conv6')
        x = instance_norm(x, 'IN3')
        x = leaky_relu(x, 0.2)

        x = conv(x, fmaps//64, kernel_size=3, stride=1, scope='conv7')

        return x
#-----------------------------------------------------------------------------------