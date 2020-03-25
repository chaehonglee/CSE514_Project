# <annie.lee@wustl.edu>
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

WEIGHT_DECAY = 1e-4
BATCH_SIZE = 4
EPSILON = 1e-6


def conv(net, in_layer, out_shape, name, ksz=[3, 3], strides=[1, 1], transpose=False, bn=True, relu=True):
    sq = tf.sqrt(1. / (ksz[0] * ksz[1] * int(in_layer.get_shape()[-1])))

    wname = name + 'conv'
    if transpose:
        if wname in net.weights.keys():
            w = net.weights[wname]
        else:
            w = tf.Variable(tf.truncated_normal([ksz[0], ksz[1], out_shape, int(in_layer.get_shape()[-1])], stddev=sq))
            net.weights[wname] = w
        output_shape = [BATCH_SIZE, int(in_layer.get_shape()[1]) * strides[0],
                        int(in_layer.get_shape()[2]) * strides[1], out_shape]
        l = tf.nn.conv2d_transpose(in_layer, w, output_shape, [1, strides[0], strides[1], 1], name=wname)
    else:
        if wname in net.weights.keys():
            w = net.weights[wname]
        else:
            w = tf.Variable(tf.truncated_normal([ksz[0], ksz[1], int(in_layer.get_shape()[-1]), out_shape], stddev=sq))
            net.weights[name + 'conv'] = w
        l = tf.nn.conv2d(in_layer, w, [1, strides[0], strides[1], 1], padding='SAME', name=name + 'conv')

    if bn:
        mean, var = tf.nn.moments(l, axes=[0, 1, 2])
        net.batchnorm_vals[name + 'BN_mean'] = mean
        net.batchnorm_vals[name + 'BN_var'] = var
        l = tf.nn.batch_normalization(l, mean, var, None, None, EPSILON, name + 'BN')

    if relu:
        bname = name + 'bias'
        if bname in net.weights.keys():
            bias = net.weights[bname]
        else:
            bias = tf.Variable(tf.zeros([out_shape]))
            net.weights[bname] = bias
        l = l + bias
        l = tf.nn.relu(l)

    return l


def unet(net, in_layer, channels, nlev, name):
    idx = nlev - 1

    r = conv(net, in_layer, channels[idx], name + '/down/1/')
    r = conv(net, r, channels[idx], name + '/down/2/')

    if nlev == 1:
        return r

    l = tf.nn.max_pool(r, [1, 2, 2, 1], [1, 2, 2, 1], padding='VALID')

    l = unet(net, l, channels, nlev - 1, 'lvl%d' % (nlev - 1))

    l = conv(net, l, channels[idx], name + '/up/1/', strides=[2, 2], transpose=True)
    l = tf.concat([r, l], axis=-1)
    l = conv(net, l, channels[idx], name + '/up/2/')

    return l


class Net:
    def __init__(self):
        self.weights = {}
        self.batchnorm_vals = {}

    def build_model(self, img, gt):
        self._build_architecture(img)

        # Calculate Loss
        self.loss = tf.reduce_mean(tf.sqrt(tf.reduce_mean((gt - self.prediction) ** 2, axis=[0, 1, 2])))
        self.loss += WEIGHT_DECAY * tf.add_n(
            [tf.nn.l2_loss(tf.cast(v, tf.float32)) for v in tf.trainable_variables()])

        # Calculate Accuracy
        self.epe = tf.reduce_mean(tf.sqrt(tf.reduce_mean((gt - self.prediction) ** 2, axis=[0, 1, 2])))
        self.rmse = tf.sqrt(tf.reduce_mean(tf.abs(gt - self.prediction) ** 2))
        self.L1 = tf.reduce_mean(tf.abs(gt - self.prediction))

        # Save image
        # saveim = self.prediction * tf.constant(64., tf.float32) + tf.constant(2. ** 15, tf.float32)
        # saveim = tf.maximum(tf.minimum(saveim, tf.constant(2 ** 16 - 1, dtype=tf.float32)), tf.constant(0., tf.float32))
        # saveim = tf.split(saveim, [1, 1, 1, 1], axis=0)
        # saveim = tf.concat([saveim[0], saveim[1], saveim[2]], axis=1)
        # saveim = tf.reshape(tf.cast(saveim, tf.uint16), [saveim.get_shape()[1] * 2, saveim.get_shape()[2], 1])
        # self.image = tf.image.encode_png(saveim)

        return self.prediction

    def _build_architecture(self, img):
        self.prediction = []

        # normalize input image:
        input_layer = img / tf.constant(128., tf.float32) - tf.constant(1., tf.float32)

        # UNet:
        nlev = 5
        channels = [512, 256, 128, 64, 32]
        l = unet(self, input_layer, channels, nlev, 'lvl%d' % nlev)

        # l = conv(self, tf.concat([l, pred], axis=-1), 16, 'init/decode0/', ksz=[1, 1], bn=False)
        # l = conv(self, tf.concat(l, axis=-1), 8, 'init/decode1/', ksz=[1, 1], bn=False)
        # pred = conv(self, l, 2, 'init/decode2/', bn=False, relu=False)

        # self.prediction = pred