import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import scipy.io as scio
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '3'
config = tf.ConfigProto()

config.gpu_options.allow_growth = True

session = tf.Session(config=config)


def genterator(noise, batch_size, reuse=False, trainable=True):
    with tf.variable_scope("generator", reuse=reuse):
        # layer1 FC  [-1, 96] -> [-1, 4, 4, 1024]
        f = tf.layers.dense(noise, units=4 * 4 * 1024)
        f = tf.reshape(f, [batch_size, 4, 4, 1024])
        f = tf.layers.batch_normalization(f, trainable=trainable)
        f = tf.nn.relu(f)

        # layer2 conv2d_transpose [-1, 4, 4, 1024] -> [-1, 8, 8, 512]
        dconv1 = tf.layers.conv2d_transpose(f, filters=512,
                                            kernel_size=[5, 5], strides=[2, 2], padding='SAME')
        dconv1 = tf.layers.batch_normalization(dconv1, trainable=trainable)
        dconv1 = tf.nn.relu(dconv1)

        # layer3 conv2d_transpose [-1, 8, 8, 512] -> [-1, 16, 16, 256]
        dconv2 = tf.layers.conv2d_transpose(dconv1, filters=256,
                                            kernel_size=[5, 5], strides=[2, 2], padding='SAME')
        dconv2 = tf.layers.batch_normalization(dconv2, trainable=trainable)
        dconv2 = tf.nn.relu(dconv2)

        # layer4 conv2d_transpose [-1, 16, 16, 256] -> [-1, 32, 32, 128]
        dconv3 = tf.layers.conv2d_transpose(dconv2, filters=128,
                                            kernel_size=[5, 5], strides=[2, 2], padding='SAME')
        dconv3 = tf.layers.batch_normalization(dconv3, trainable=trainable)
        dconv3 = tf.nn.relu(dconv3)

        # layer5 conv2d_transpose [-1, 32, 32, 128] -> [-1, 64, 64, 3]
        dconv4 = tf.layers.conv2d_transpose(dconv3, filters=3,
                                            kernel_size=[5, 5], strides=[2, 2], padding='SAME')
        dconv4 = tf.layers.batch_normalization(dconv4, trainable=trainable)
        dconv4 = tf.nn.tanh(dconv4)

        return dconv4


def discriminator(image, batch_size, reuse=False, trainable=True):
    with tf.variable_scope("discriminator", reuse=reuse):
        # layer1 conv2d [-1, 64, 64, 3] -> [-1, 32, 32, 64]
        conv1 = tf.layers.conv2d(image, filters=64,
                                 kernel_size=[5, 5], strides=[2, 2], padding='SAME')
        conv1 = tf.layers.batch_normalization(conv1, trainable=trainable)
        conv1 = tf.nn.leaky_relu(conv1, alpha=0.2)

        # layer2 conv2d [-1, 32, 32, 64] -> [-1, 16, 16, 128]
        conv2 = tf.layers.conv2d(conv1, filters=128,
                                 kernel_size=[5, 5], strides=[2, 2], padding='SAME')
        conv2 = tf.layers.batch_normalization(conv2, trainable=trainable)
        conv2 = tf.nn.leaky_relu(conv2, alpha=0.2)

        # layer3 conv2d [-1, 16, 16, 128]  -> [-1, 8, 8, 256]
        conv3 = tf.layers.conv2d(conv2, filters=256,
                                 kernel_size=[5, 5], strides=[2, 2], padding='SAME')
        conv3 = tf.layers.batch_normalization(conv3, trainable=trainable)
        conv3 = tf.nn.leaky_relu(conv3, alpha=0.2)

        # layer4 conv2d [-1, 8, 8, 256] -> [-1, 4, 4, 512]
        conv4 = tf.layers.conv2d(conv3, filters=512,
                                 kernel_size=[5, 5], strides=[2, 2], padding='SAME')
        conv4 = tf.layers.batch_normalization(conv4, trainable=trainable)
        conv4 = tf.nn.leaky_relu(conv4, alpha=0.2)

        # layer5 FC [-1, 4*4*512] -> [-1, 1]
        conv5 = tf.layers.flatten(conv4)
        conv5 = tf.layers.dense(conv5, units=1)

        return conv5


def showImage(images):
    fig = plt.figure(figsize=(8, 8))
    gs = gridspec.GridSpec(8, 8)
    gs.update(wspace=0, hspace=0)

    for i, sample in enumerate(images):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample.reshape([64, 64, 3]))
    plt.show()


def train(image, noise_dim=96, iter_epoch=500, batch_size=128, learning_rate=0.0002, beta1=0.5):
    tf.reset_default_graph()
    x = tf.placeholder(tf.float32, shape=[None, 64, 64, 3])
    noise = tf.placeholder(tf.float32, shape=[None, noise_dim])
    G = genterator(noise=noise, batch_size=batch_size, reuse=False)
    D_real = discriminator(x, batch_size=batch_size, reuse=False)
    D_fake = discriminator(G, batch_size=batch_size, reuse=True)

    D_real_loss = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=D_real, labels=tf.ones_like(D_real)))
    D_fake_loss = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake, labels=tf.zeros_like(D_fake)))
    D_loss = D_real_loss + D_fake_loss

    G_loss = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake, labels=tf.ones_like(D_fake)))

    t_vars = tf.trainable_variables()
    d_vars = [var for var in t_vars if var.name.startswith('discriminator')]
    g_vars = [var for var in t_vars if var.name.startswith('generator')]
    with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
        g_op = tf.train.AdamOptimizer(learning_rate=learning_rate,
                                      beta1=beta1).minimize(G_loss, var_list=g_vars)
        d_op = tf.train.AdamOptimizer(learning_rate=learning_rate,
                                      beta1=beta1).minimize(D_loss, var_list=d_vars)

    print("network build success")

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for i in range(iter_epoch):
            total_batch = int(image.shape[0] / batch_size)

            for j in range(total_batch):
                mini_batch = image[j * batch_size:(j + 1) * batch_size]
                ns1 = np.random.uniform(low=-1, high=1, size=[batch_size, noise_dim])
                ds, _ = sess.run([D_loss, d_op], feed_dict={x: mini_batch, noise: ns1})

                ns2 = np.random.uniform(low=-1, high=1, size=[batch_size, noise_dim])
                gs, _ = sess.run([G_loss, g_op], feed_dict={x: mini_batch, noise: ns2})
            print(i + 1)
            ns2 = np.random.uniform(low=-1, high=1, size=[batch_size, noise_dim])
            g_img = sess.run(G, feed_dict={noise: ns2})
            g_img = (g_img[:64] + 1) * 127.5
            g_img = tf.cast(g_img, tf.int32)
            g_img = sess.run(g_img)
            showImage(g_img)


def data2train():
    data = scio.loadmat("Anime.mat")
    images = np.array(data['imageData']).astype(np.float)
    # 处理数据 将数据收缩到[-1, 1]
    images = images / 127.5 - 1
    print(images.shape)
    train(images)


if __name__ == '__main__':
    data2train()
