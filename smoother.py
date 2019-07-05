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
flags.DEFINE_integer('kernel_size', 5, 'kernel size, default: 5')
#-----------------------------------------------------------------------------------
# 主进程
dataset_path = './dataset/'
dataset_path = os.path.join(dataset_path, FLAGS.dataset)

trainB_path = os.path.join(dataset_path, 'trainB')

trainB_smooth_path = os.path.join(dataset_path, 'trainB_smooth')
check_logdir(trainB_smooth_path)

list_trainB = glob_dataset(trainB_path)

counter, length = 1, len(list_trainB)

for path in list_trainB:
    img = cv2.imread(path, 1)
    img = cv2.resize(img, (FLAGS.img_size, FLAGS.img_size))
    img = cv2.blur(img, (FLAGS.kernel_size, FLAGS.kernel_size))

    filename = path.split('trainB')[-1]
    save_path = os.path.join(trainB_smooth_path, filename[1:])

    cv2.imwrite(save_path, img)
    print('DONE: %d/%d'%(counter, length))
    counter += 1
#-----------------------------------------------------------------------------------