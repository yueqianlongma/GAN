##
#    2019/11/6
#    第一次学习GAN(生成对抗网络)
#    propose: 创建GAN网络，  生成minist图片
#
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.gridspec as gridspec
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '3'
config = tf.ConfigProto()

config.gpu_options.allow_growth = True

session = tf.Session(config=config)

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

BATCH_SIZE = 128
IMAGE_SIZE = 28 * 28
PG_SIZE = 100
learning_rate = 0.01
iter_epoch = 100000
keep_prob = 0.3

def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
    return tf.random_normal(shape=size, stddev=xavier_stddev)

def random_data(shape):
    return np.random.uniform(-1., 1, shape)

Dw_1 = tf.Variable(xavier_init([784, 128]))
Db_1 = tf.Variable(tf.zeros(shape=[128]))
Dw_2 = tf.Variable(xavier_init([128, 1]))
Db_2 = tf.Variable(tf.zeros(shape=[1]))
theta_D = [Dw_1, Db_1, Dw_2, Db_2]

Gw_1 = tf.Variable(xavier_init([100, 128]))
Gb_1 = tf.Variable(tf.zeros(shape=[128]))
Gw_2 = tf.Variable(xavier_init([128, 784]))
Gb_2 = tf.Variable(tf.zeros(shape=[784]))
theta_G = [Gw_1, Gb_1, Gw_2, Gb_2]


def D(z):
    layer1 = tf.nn.relu(tf.matmul(z, Dw_1) + Db_1)
    output = tf.matmul(layer1, Dw_2) + Db_2
    return output, tf.nn.sigmoid(output)

def G(z):
    layer1 = tf.nn.relu(tf.matmul(z, Gw_1) + Gb_1)
    output = tf.nn.sigmoid(tf.matmul(layer1, Gw_2) + Gb_2)
    return output

def showImage(images):
    fig = plt.figure(figsize=(4, 4))
    gs = gridspec.GridSpec(4, 4)
    gs.update(wspace=0.05, hspace=0.05)

    for i, sample in enumerate(images):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample.reshape(28, 28), cmap='Greys_r')
    plt.show()


def train():
    x = tf.placeholder(tf.float32, shape=[None, IMAGE_SIZE])
    pg = tf.placeholder(tf.float32, shape=[None, PG_SIZE])

    g_image = G(pg)
    d_net_real, d_net_real_prob = D(x)
    d_net_fake, d_net_fake_prob = D(g_image)

    # d_loss = -tf.reduce_mean(tf.reduce_sum(tf.log(d_net_real) + tf.log(1 - d_net_fake), axis=1), axis=0)
    # g_loss = -tf.reduce_mean(tf.reduce_sum(tf.log(d_net_fake), axis=1), axis=0)
    D_loss_real = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=d_net_real, labels=tf.ones_like(d_net_real_prob)))
    D_loss_fake = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=d_net_fake, labels=tf.zeros_like(d_net_fake_prob)))
    D_loss = D_loss_real + D_loss_fake
    G_loss = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=d_net_fake, labels=tf.ones_like(d_net_fake_prob)))

    d_optimizer = tf.train.AdamOptimizer().minimize(D_loss, var_list=theta_D)
    g_optimizer = tf.train.AdamOptimizer().minimize(G_loss, var_list=theta_G)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        ls_d, ls_g = [], []
        for i in range(iter_epoch):
            noise = random_data([BATCH_SIZE, PG_SIZE])
            image, _ = mnist.train.next_batch(BATCH_SIZE)
            loss_d, _ = sess.run([D_loss, d_optimizer], feed_dict={x: image, pg: noise})
            loss_g, _ = sess.run([G_loss, g_optimizer], feed_dict={x: image, pg: noise})

            if i % 1000 == 0:
                g_images = sess.run(g_image, feed_dict={pg: random_data([16, PG_SIZE])})
                showImage(g_images)

                print('Iter: {}'.format(i))
                print('D loss: {:.4}'.format(loss_d))
                print('G_loss: {:.4}'.format(loss_g))
                print()


if __name__ == '__main__':
    train()
