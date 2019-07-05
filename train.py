from networks import *
from utils import *
from loss import *
from vgg19 import Vgg19
from datetime import datetime
from random import shuffle

import tensorflow as tf
import numpy as np
import glob
import os
import sys
#-----------------------------------------------------------------------------------
# 超参数输入口
flags = tf.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('save_model_path', './model/', 'the save path for the model')
flags.DEFINE_string('load_model_path', None, 'the loading path of the model')
flags.DEFINE_string('outfile_path', './outfile/', 'the save path for the visual files')
flags.DEFINE_string('trainA', './dataset/person2cartoon/trainA', 'the path of the training domainA')
flags.DEFINE_string('trainB', './dataset/person2cartoon/trainB', 'the path of the training domainB')
flags.DEFINE_string('trainB_smooth', './dataset/person2cartoon/trainB_smooth', 'the path of the blured domainB')
flags.DEFINE_string('GAN_TYPE', 'GAN', 'Whether to use gradient penalty')

flags.DEFINE_integer('img_size', 256, 'image size, default: 256')
flags.DEFINE_integer('save_image_step', 100, 'save the image every 100 steps')
flags.DEFINE_integer('save_summary_step', 100, 'save the summary every 100 steps')
flags.DEFINE_integer('total_epoch', 10, 'Total training epochs, default: 10')

flags.DEFINE_float("l1_lambda", 10, 'a weight of l1_loss, default: 10')
flags.DEFINE_float('gp_lambda', 10, 'a weight of gradient penalty, default: 10')
flags.DEFINE_float('vgg_weight', 10, 'a weight of vgg_loss, default: 10')
flags.DEFINE_float('adv_weight', 1, 'a weight of GAN, default: 10')
flags.DEFINE_float('learning_rate', 2e-4, 'the learning rate of train, default: 0.0002')
#-----------------------------------------------------------------------------------
# 主进程
def main():
    # 确认是否为新训练
    if not FLAGS.load_model_path:
        model_path = nowtime_dir(FLAGS.save_model_path)
        outfile_path = os.path.join(FLAGS.outfile_path, model_path.split('/')[-1])
        check_logdir(outfile_path)
    else:
        model_path = FLAGS.load_model_path
        outfile_path = os.path.join(FLAGS.outfile_path, model_path.split('/')[-1])

    # 输入占位，x为目标域，y为源域，bx为x辅助域
    lr = tf.placeholder(tf.float32, None, name='learning_rate')
    real_a = tf.placeholder(tf.float32, shape=[1, FLAGS.img_size, FLAGS.img_size, 3], name='trainA')
    real_b = tf.placeholder(tf.float32, shape=[1, FLAGS.img_size, FLAGS.img_size, 3], name='trainB')
    real_bs = tf.placeholder(tf.float32, shape=[1, FLAGS.img_size, FLAGS.img_size, 3], name='trainB_smooth')

    fake_b = generator(real_a)

    d_real_b = discriminator(x=real_b)
    d_fake_b = discriminator(x=fake_b, reuse=True)
    d_real_bs = discriminator(x=real_bs, reuse=True)

    # 梯度惩罚
    if FLAGS.GAN_TYPE == 'WGAN':
        GP = gradient_penalty(real_b, fake_b)+gradient_penalty(real_b, real_bs)
    else:
        GP = 0.

    # 定义损失
    loss_v = FLAGS.vgg_weight*vgg_loss(real_a, fake_b)
    loss_g = FLAGS.adv_weight*G_loss(d_fake_b)

    loss_generator = loss_v+loss_g
    loss_discriminator = FLAGS.adv_weight*D_loss(d_real_b, d_fake_b, d_real_bs)+FLAGS.gp_lambda*GP

    # 训练参数
    variables = tf.trainable_variables()
    vars_G = [v for v in variables if 'generator' in v.name]
    vars_D = [v for v in variables if 'discriminator' in v.name]

    # 定义学习器
    adam = tf.train.AdamOptimizer(lr, beta1=0.5, beta2=0.999)

    optim_init = adam.minimize(loss_v, var_list=vars_G)
    optim_G = adam.minimize(loss_generator, var_list=vars_G)
    optim_D = adam.minimize(loss_discriminator, var_list=vars_D)

    # 日志
    sum_vgg = tf.summary.scalar('vgg loss', loss_v)
    sum_generator = tf.summary.scalar('generator loss', loss_generator)
    sum_discriminator = tf.summary.scalar('discriminator', loss_discriminator)

    # 新建会话
    sess = tf.Session()
    summary_writer = tf.summary.FileWriter(model_path, graph=sess.graph)
    model_saver = tf.train.Saver()

    # 初始化
    if not FLAGS.load_model_path:
        sess.run(tf.global_variables_initializer())

        step_now = 0

    else:
        checkpoint = tf.train.get_checkpoint_state(model_path)
        metapath = checkpoint.model_checkpoint_path+'.meta'
        restore = tf.train.import_meta_graph(metapath)
        restore.restore(sess, tf.train.latest_checkpoint(model_path))

        step_now = int(metapath.split('-')[-1].split('.')[0])

    list_a = glob_dataset(FLAGS.trainA)
    list_b = glob_dataset(FLAGS.trainB)
    list_bs = glob_dataset(FLAGS.trainB_smooth)
    list_a, list_b, list_bs = alignment_data(list_a, list_b, list_bs)

    length = len(list_a)

    init_lr = FLAGS.learning_rat*pow(0.9, step_now//length)

    # 训练开始
    for step in range(step_now, FLAGS.total_epoch*length):
        instance_a = imreader(list_a, step, FLAGS.img_size)
        instance_b = imreader(list_b, step, FLAGS.img_size)
        instance_bs = imreader(list_bs, step, FLAGS.img_size)

        instance_a = expand_dims(instance_a)
        instance_b = expand_dims(instance_b)
        instance_bs = expand_dims(instance_bs)

        if step%length == 0 and step != 0:
            model_save(model_saver, sess, step-1, model_path)

            init_lr *= 0.9
            print('learning rate: {}'.format(init_lr))

            shuffle(list_a), shuffle(list_b), shuffle(list_bs)

        feed_dict = {lr: init_lr, real_a: instance_a, real_b: instance_b, real_bs: instance_bs}

        if step < length:
            _, loss_value_vgg = sess.run([optim_init, loss_v], feed_dict=feed_dict)

            if (step+1)%FLAGS.save_summary_step == 0:
                print('step: %7d, vgg_loss: %.3f'%(step+1, loss_value_vgg))

                sum_value_vgg = sess.run(sum_vgg, feed_dict=feed_dict)
                summary_writer.add_summary(sum_value_vgg, step+1)

        else:
            _, loss_value_d = sess.run([optim_D, loss_discriminator], feed_dict=feed_dict)

            _, loss_value_g = sess.run([optim_G, loss_generator], feed_dict=feed_dict)

            if (step+1)%FLAGS.save_summary_step == 0:
                print('step: %7d, generator_loss: %.3f, discriminator_loss: %.3f'%(step+1, loss_value_g, loss_value_d))

                sum_value_g, sum_value_d = sess.run([sum_generator, sum_discriminator], feed_dict=feed_dict)
                summary_writer.add_summary(sum_value_g, step+1)
                summary_writer.add_summary(sum_value_d, step+1)

        if (step+1)%FLAGS.save_image_step == 0:
            fake_image = sess.run(fake_b, feed_dict=feed_dict)
            imsave(instance_a[0], fake_image[0], step+1, outfile_path)
#-----------------------------------------------------------------------------------
# 运行点
if __name__ == '__main__':
    main()
#-----------------------------------------------------------------------------------