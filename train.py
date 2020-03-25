# <annie.lee@wustl.edu>
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import numpy as np
import tensorflow as tf

import utils as ut
from model import Net
from data import Dataset

DATA = ''
LEARNING_RATE = 1e-3
BATCH_SIZE = 4
MAXITER = 500e3
SAVEITER = 1e4
DISPITER = 10
VALITER = 1000
VALREP = 2
SAVEIMAGE = 1e4

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# Load saver
saver = ut.ckpter('wts/model*.npz')
if saver.iter >= MAXITER:
    MAXITER = 550e3
    LEARNING_RATE = 1e-5

if saver.iter >= MAXITER:
    MAXITER = 600e3
    LEARNING_RATE = 1e-6

# Load dataset
d = Dataset(BATCH_SIZE)

# Build graph
model = Net()
output = model.build_model(d.img, d.ground_truth)

# Define Optimizer
optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE)
train_op = optimizer.minimize(model.loss, var_list=list(model.weights.values()))

# Start session
sess = tf.Session(config=tf.ConfigProto(intra_op_parallelism_threads=4))
sess.run(tf.global_variables_initializer())

# Load data file names
train_img = [f.rstrip('\n') for f in open('../data/' + DATA + '.txt').readlines()]
val_list = [f.rstrip('\n') for f in open('../data/' + DATA + '.txt').readlines()]
ESIZE = len(train_img) // BATCH_SIZE
VESIZE = len(val_list) // BATCH_SIZE

# Setup save/restore
origiter = saver.iter
rs = np.random.RandomState(0)
if origiter > 0:
    ut.loadNet(saver.latest, model, sess)
    if os.path.isfile('wts/opt.npz'):
        ut.loadAdam('wts/opt.npz', optimizer, model.weights, sess)

    for k in range((origiter + ESIZE - 1) // ESIZE):
        idx = rs.permutation(len(train_img))
    ut.mprint("Restored to iteration %d" % origiter)

# Main Training Loop
niter = origiter
touts = 0.
while niter < MAXITER + 1:

    # Validation for current training status
    if niter % VALITER == 0:
        vouts = 0.
        for j in range(VALREP):
            off = j % (len(val_list) % BATCH_SIZE + 1)
            for b in range(VESIZE):
                blst = val_list[(b * BATCH_SIZE + off):((b + 1) * BATCH_SIZE + off)]
                outs = sess.run([model.loss, model.L1_3, model.epe], feed_dict=d.fdict(blst))
                vouts = vouts + np.float32(outs)

        if niter % SAVEIMAGE == 0:
            # code for saving progress images
            pass

        vouts = vouts / np.float32(VESIZE * VALREP)
        ut.vprint(niter, ['loss.v', 'L1.v', 'epe.v'], vouts)

    if niter == MAXITER:
        break

    if niter % ESIZE == 0:
        idx = rs.permutation(len(train_img))

    blst = [train_img[idx[(niter % ESIZE) * BATCH_SIZE + b]] for b in range(BATCH_SIZE)]
    _, outs = sess.run([train_op, [model.loss, model.L1_3, model.epe]], feed_dict=d.fdict(blst))
    niter = niter + 1
    touts = touts + np.float32(outs)

    if niter % SAVEITER == 0:
        ut.saveNet('wts/model_%d.npz' % niter, model, sess)
        saver.clean(every=SAVEITER, last=1)
        ut.mprint('Saved Model')

    if niter % DISPITER == 0:
        touts = touts / np.float32(DISPITER)
        ut.vprint(niter, ['lr', 'loss.t', 'L1.t', 'epe.t'], [LEARNING_RATE] + list(touts))
        touts = 0.
        if ut.stop:
            break

if niter > saver.iter:
    ut.saveNet('wts/model_%d.npz' % niter, model, sess)
    saver.clean(every=SAVEITER, last=1)
    ut.mprint('Saved Model')

if niter > origiter:
    ut.saveAdam('wts/opt.npz', optimizer, model.weights, sess)
    ut.mprint("Saved Optimizer.")
