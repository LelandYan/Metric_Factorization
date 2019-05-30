# _*_ coding: utf-8 _*_
__author__ = 'LelandYan'
__date__ = '2019/5/30 16:52'
import numpy as np
import itertools
import tensorflow as tf
# def a():
#     res = itertools.combinations([1, 2, 3, 4], 2)
#     print("@@",list(res))
# res = itertools.combinations(np.arange(3),2)
# print(list(res))
input1 = tf.placeholder(tf.float32,shape=[None])
input2 = tf.placeholder(tf.float32)
a =input1.shape
print(a)
# 定义乘法运算
output = tf.multiply(input1[0], input2)

# 通过session执行乘法运行
with tf.Session() as sess:
    # 执行时要传入placeholder的值

    print(sess.run(output, feed_dict = {input1:[7.,2], input2: [2.]}))