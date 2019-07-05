from utils import check_logdir, glob_dataset

import os
import glob
import cv2
import tensorflow as tf
#-----------------------------------------------------------------------------------
# 超参数入口
flags = tf.flags
FLAGS = tf.flags.FLAGS

flags.DEFINE_string('dataset', 'person2cartoon', 'the path of the training domainB')

flags.DEFINE_integer('img_size', 256, 'image size, default: 256')
#-----------------------------------------------------------------------------------
# 主进程
dataset_path = './dataset/'
dataset_path = os.path.join(dataset_path, FLAGS.dataset)

trainB_path = os.path.join(dataset_path, 'trainB')

trainB_smooth_path = os.path.join(dataset_path, 'trainB_smooth')
check_logdir(trainB_smooth_path)

list_trainB = glob_dataset(trainB_path)

for path in list_trainB:
    img_path = path
    img = cv2.imread(img_path, 1)
    img = cv2.resize(img, (FLAGS.img_size, FLAGS.img_size))
    img = cv2.blur(img, (3, 3))

    filename = path.split('trainB')[-1]
    trainB_smooth_path = os.path.join(trainB_smooth_path, filename[2:])

    cv2.imwrite(trainB_smooth_path, img)
#-----------------------------------------------------------------------------------