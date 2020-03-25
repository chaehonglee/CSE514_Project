# <annie.lee@wustl.edu>
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf


class Dataset:
    def __init__(self,
                 batch_size,
                 imstr='/path_to_image/%s',
                 gtstr='/path_to_ground_truth/%s'):

        self.batch_size = batch_size
        self.imstr = imstr
        self.gtstr = gtstr

        self.graph()

    def fdict(self, ids):
        fd = {}
        assert len(ids) == self.batch_size
        for i in range(self.batch_size):
            fd[self.img_placeholder[i]] = self.imstr % ids[i]
            fd[self.gt_placeholder[i]] = self.gtstr % ids[i]
        return fd

    def graph(self):
        self.img_placeholder = []
        self.gt_placeholder = []

        # Create placeholders for image names
        for i in range(self.batch_size):
            self.img_placeholder.append(tf.placeholder(tf.string))
            self.gt_placeholder.append(tf.placeholder(tf.string))

        # Parse and augment images
        self.img, self.ground_truth = self._parse(self.img_placeholder, self.gt_placeholder)
        self.img, self.ground_truth = self._augment(self.img, self.ground_truth)

    @staticmethod
    def _parse(img, ground_truth):
        """

        Parses file names and processes images to return:

        :return: list of image tensor (w, h, 3),
                 list of interpolated sparse flow (w, h, 2),
                 list of true optical flow (w, h, 2)
                 list of invalid flow points as mask (w, h, 1)
        """
        _img = []
        _ground_truth = []

        for i in range(len(img)):
            im = tf.image.decode_png(tf.read_file(img[i]), dtype=tf.uint8, channels=3)
            _img.append(tf.cast(im, tf.float32))

            gt = tf.image.decode_png(tf.read_file(ground_truth[i]), dtype=tf.uint16, channels=1)
            gt = tf.cast(gt, tf.float32)
            gt = tf.divide(tf.add(gt, tf.constant(-2 ** 15, dtype=tf.float32)), tf.constant(64, dtype=tf.float32))
            gt = tf.reshape(gt, [tf.shape(im)[0], tf.shape(im)[1], 2])
            _ground_truth.append(gt)

        return _img, _ground_truth

    @staticmethod
    def _augment(img, ground_truth):
        """
        Helper for applying augmentation on an (image, label) pair
        """
        for i in range(len(img)):
            im = tf.image.random_brightness(img, max_delta=0.2)
            im = tf.image.random_contrast(im, 0.8, 1.2)
            im = tf.image.random_saturation(im, 0.8, 1.2)

            concat = tf.concat([im, ground_truth[i]], axis=-1)

            isflip = tf.random_uniform((), 0, 2, dtype=tf.int32)
            concat = tf.cond(isflip > 1, lambda: tf.image.random_flip_left_right(concat), lambda: concat)

            yc = tf.random_uniform((), 0, tf.shape(img[i])[0] - 320 + 1, dtype=tf.int32)
            xc = tf.random_uniform((), 0, tf.shape(img[i])[1] - 1024 + 1, dtype=tf.int32)
            concat = tf.slice(concat, tf.stack([yc, xc, 0]), tf.stack([320, 1024, tf.shape(concat)[-1]]))

            im, gt = tf.split(concat, [3, 3], axis=-1)

            img[i] = im
            ground_truth[i] = gt

        return img, ground_truth
