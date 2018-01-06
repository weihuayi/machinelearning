import os
import math
import psutil
import socket
import subprocess

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data as mnist_data


tf.set_random_seed(0)
hidden_layer = [200, 100, 60, 20]
layer = [784] + hidden_layer + [10]
mnist = mnist_data.read_data_sets("./data/input_data", one_hot=True, reshape=False, validation_size=0)


def logs_clear():
    if 'logs' in os.listdir('./'):
        subprocess.Popen('rm -rf *', shell=True, cwd='./logs')
    else:
        subprocess.Popen('mkdir logs', shell=True)


def kill_tensorboard_process():
    for pid in psutil.pids():
        try:
            cmd = psutil.Process(pid=pid)
            if cmd.name() == 'tensorboard':
                subprocess.call(['kill %d' % pid], shell=True)
        except psutil.NoSuchProcess:
            pass


def variable_summaries(var):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)


with tf.name_scope('input'):
    X = tf.placeholder(tf.float32, [None, 28, 28, 1], name='X')
    Y_ = tf.placeholder(tf.float32, [None, 10], name='Y_')
    X_0 = tf.reshape(X, [-1, 28*28])

with tf.name_scope('param'):
    lr = tf.placeholder(tf.float32, name='learning_rate')
    keep_prob = tf.placeholder(tf.float32, name='probability_of_keeping')

for i in range(len(hidden_layer)):
    with tf.name_scope('hidden_%d' % i):
        with tf.name_scope('weight'):
            exec('W_%d = tf.Variable(tf.truncated_normal([layer[%d], layer[%d + 1]], stddev=0.1))' % (i, i, i))
            exec('variable_summaries(var=W_%d)' % i)
        with tf.name_scope('bias'):
            exec('B_%d = tf.Variable(tf.ones([layer[%d + 1]]) / 10)' % (i, i))
            exec('variable_summaries(var=B_%d)' % i)
        with tf.name_scope('relu'):
            exec('R_%d = tf.nn.relu(tf.matmul(X_%d, W_%d) + B_%d)' % (i, i, i, i))
        with tf.name_scope('drop'):
            exec('X_%d = tf.nn.dropout(R_%d, keep_prob)' % (i + 1, i))

with tf.name_scope('output'):
    with tf.name_scope('weight'):
        exec('W_%d = tf.Variable(tf.truncated_normal([layer[%d], layer[%d + 1]], stddev=0.1))' % (len(hidden_layer), len(hidden_layer), len(hidden_layer)))
        exec('variable_summaries(var=W_%d)' % len(hidden_layer))
    with tf.name_scope('bias'):
        exec('B_%d = tf.Variable(tf.zeros([layer[%d + 1]]))' % (len(hidden_layer), len(hidden_layer)))
        exec('variable_summaries(var=B_%d)' % len(hidden_layer))
    with tf.name_scope('Y_logits'):
        Y_logits = eval('tf.matmul(X_%d, W_%d) + B_%d' % (len(hidden_layer), len(hidden_layer), len(hidden_layer)))
    with tf.name_scope('Y'):
        Y = tf.nn.softmax(Y_logits)

with tf.name_scope('loss'):
    with tf.name_scope('cross_entropy'):
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=Y_logits, labels=Y_)
    with tf.name_scope('loss'):
        loss = tf.reduce_mean(cross_entropy)
        tf.summary.histogram('loss', loss)

with tf.name_scope('accuracy'):
    with tf.name_scope('correct_prediction'):
        correct_prediction = tf.equal(tf.argmax(Y, 1), tf.argmax(Y_, 1))
    with tf.name_scope('accuracy'):
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.summary.histogram('accuracy', accuracy)

train_step = tf.train.AdamOptimizer(lr).minimize(loss)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

logs_clear()
merged = tf.summary.merge_all()
writer = tf.summary.FileWriter('./logs', tf.get_default_graph())

max_learning_rate = 0.003
min_learning_rate = 0.0001
decay_speed = 2000.0

for i in range(500):
    batch_X, batch_Y = mnist.train.next_batch(100)
    learning_rate = min_learning_rate + (max_learning_rate - min_learning_rate) * math.exp(-i/decay_speed)

    a, c = sess.run([accuracy, loss], {X: batch_X, Y_: batch_Y, keep_prob: 1.0})
    print(str(i) + ": accuracy:" + str(a) + " loss: " + str(c) + " (lr:" + str(learning_rate) + ")")

    a, c = sess.run([accuracy, loss], {X: mnist.test.images, Y_: mnist.test.labels, keep_prob: 1.0})
    print(str(i) + ":" + str(i*100//mnist.train.images.shape[0]+1) + ",\ttest accuracy:" + str(a) + ",\ttest loss: " + str(c))

    summary, _ = sess.run([merged, train_step], {X: batch_X, Y_: batch_Y, keep_prob: 0.75, lr: learning_rate})
    writer.add_summary(summary, i)

writer.close()
kill_tensorboard_process()
p = subprocess.Popen("tensorboard --logdir='./logs'", shell=True)
subprocess.Popen('firefox http://%s:%d/#graphs' % (socket.gethostname(), 6006), shell=True)
