from networks import discriminator
from vgg19 import Vgg19

import tensorflow as tf
import numpy as np
#-----------------------------------------------------------------------------------
# L1 loss
def L1_loss(A, B):
    return tf.reduce_mean(abs(A-B))
#-----------------------------------------------------------------------------------
# 生成器损失
def G_loss(fake, GAN_TYPE):
    if GAN_TYPE == 'GAN':
        loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                              labels=tf.ones_like(fake), logits=fake))

    if GAN_TYPE == 'WGAN':
        loss = -tf.reduce_mean(fake)

    if GAN_TYPE == 'LSGAN':
        loss = tf.reduce_mean((fake-tf.ones_like(fake)**2))

    return loss
#-----------------------------------------------------------------------------------
# 判别器损失
def D_loss(real, fake, blured_real, GAN_TYPE):
    if GAN_TYPE == 'GAN': 
        real_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                                   labels=tf.ones_like(real), logits=real))
        fake_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                                   labels=tf.zeros_like(fake), logits=fake))
        blured_real_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                                   labels=tf.zeros_like(blured_real), logits=blured_real))

    if GAN_TYPE == 'WGAN':
        real_loss = -tf.reduce_mean(real)
        fake_loss = tf.reduce_mean(fake)
        blured_real_loss = tf.reduce_mean(blured_real)

    if GAN_TYPE == 'LSGAN':
        real_loss = tf.reduce_mean((real-tf.ones_like(real))**2)
        fake_loss = tf.reduce_mean((fake-tf.ones_like(real))**2)
        blured_real_loss = tf.reduce_mean((blured_real-tf.ones_like(blured_real))**2)

    return real_loss+fake_loss+blured_real_loss
#-----------------------------------------------------------------------------------
# vgg损失
def vgg_loss(real, fake):
    vgg = Vgg19('vgg19.npy')

    vgg.build(real)
    real_feature_map = vgg.conv4_4_no_activation

    vgg.build(fake)
    fake_feature_map = vgg.conv4_4_no_activation

    return L1_loss(real_feature_map, fake_feature_map)
#-----------------------------------------------------------------------------------
# 梯度惩罚
def gradient_penalty(real_img, fake_image, scope='discriminator'):
    difference = real_img-fake_image
    alpha = tf.random_uniform(shape=[1,1,1,1], minval=0., maxval=1.)
    interpolates = real_img+alpha*difference
    
    gradients = tf.gradients(discriminator(interpolates, reuse=True, scope=scope), [interpolates])[0]
    slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients),reduction_indices=[1,2,3]))

    return tf.reduce_mean((slopes-1.)**2)
#-----------------------------------------------------------------------------------
