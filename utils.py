from datetime import datetime

import os
import numpy as np
import glob
import cv2
#-----------------------------------------------------------------------------------
# 文件目录检查与创建
def check_logdir(logdir):
    if not os.path.exists(logdir):
        os.makedirs(logdir)
#-----------------------------------------------------------------------------------
# 图像读取
def imreader(imlist, step, size):
    index = step%len(imlist)
    image = cv2.imread(imlist[index], 1)
    image = cv2.resize(image, (size, size))
    image = image/127.5-1.

    return image
#-----------------------------------------------------------------------------------
# 给单张图像增加batch size维度
def expand_dims(image):
    return np.expand_dims(np.array(image).astype(np.float32), axis=0)
#-----------------------------------------------------------------------------------
# 获取数据的所有名称路径
def glob_dataset(data_path):
    return glob.glob(os.path.join(data_path, '*'))
#-----------------------------------------------------------------------------------
# 对齐两个域的数据集长度，以最短为标准
def alignment_data(A, B, Bs):
    if len(A) == len(B):
        return A, B, Bs

    if len(A) < len(B):
        return A, sorted(B)[:len(A)], sorted(Bs)[:len(A)]

    return A[:len(B)], B, Bs
#-----------------------------------------------------------------------------------
# 模型存储
def model_save(saver, sess, step, logdir):
    check_logdir(logdir)

    saver.save(sess, os.path.join(logdir, 'model'), step)
    print('The checkpoint has been created: {}'.format(logdir))
#-----------------------------------------------------------------------------------
# 图片存储
def imsave(real_img, fake_img, step, save_path):
    real_img = ((real_img+1.)*127.5)
    fake_img = (fake_img+1.)*127.5

    real_img = real_img.astype(np.float32)
    fake_img = fake_img.astype(np.float32)

    result = np.concatenate((real_img, fake_img), axis=1)

    save_path = os.path.join(save_path, 'step-'+str(step)+'.png')
    cv2.imwrite(save_path, result)
#-----------------------------------------------------------------------------------
# 以当前时间创建给定目录的时间子文件夹并返回路径
def nowtime_dir(logdir):
    current_time = datetime.now().strftime('%Y%m%d-%H-%M')
    path = os.path.join(logdir, current_time)
    check_logdir(path)

    return path
#-----------------------------------------------------------------------------------