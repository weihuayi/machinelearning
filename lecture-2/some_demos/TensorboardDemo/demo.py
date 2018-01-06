import os
import math
import psutil
import socket
import subprocess

import tensorflow as tf
# noinspection SpellCheckingInspection
from tensorflow.examples.tutorials.mnist import input_data as mnist_data


class MnistDemo:
    input_data_dir = './data/input_data'
    logs_dir = './logs'
    hidden_layer = [200, 100, 60, 20]
    batch_size = 100
    max_learning_rate = 0.003
    min_learning_rate = 0.0001
    decay_speed = 2000.0

    def __init__(self, **kwargs):
        for key in kwargs:
            if hasattr(self, key):
                # TODO: We can check self.key is a method or not, but can not check it is a property or not.
                if eval("type(self.%s).__name__ == 'method'" % key):
                    raise KeyError('%s is a method in Table, not a property.' % key)
                else:
                    exec("self.%s = kwargs[%s]" % (key, key))
            else:
                raise KeyError('%s is a invalid key, Table do not have attribute named %s.' % (key, key))

        self.h_n = len(self.hidden_layer)
        # noinspection SpellCheckingInspection
        self.mnist = mnist_data.read_data_sets(self.input_data_dir, one_hot=True, reshape=False, validation_size=0)
        # noinspection PyPep8Naming
        temp_X, temp_Y = self.mnist.train.next_batch(self.batch_size)
        self.fig_size = [temp_X.shape[1], temp_X.shape[2]]
        # noinspection SpellCheckingInspection
        self.num_classifer = temp_Y.shape[1]
        self.layer = [self.fig_size[0] * self.fig_size[1]] + self.hidden_layer + [self.num_classifer]
        tf.set_random_seed(0)

        with tf.name_scope('input'):
            self.X = tf.placeholder(tf.float32, [None, self.fig_size[0], self.fig_size[1], 1], name='X')
            self.Y_ = tf.placeholder(tf.float32, [None, self.num_classifer], name='Y_')
            # noinspection PyPep8Naming,PyUnusedLocal
            X_0 = tf.reshape(self.X, [-1, self.fig_size[0]*self.fig_size[1]])

        with tf.name_scope('param'):
            self.lr = tf.placeholder(tf.float32, name='learning_rate')
            self.keep_prob = tf.placeholder(tf.float32, name='probability_of_keeping')

        for i in range(self.h_n):
            with tf.name_scope('hidden_%d' % i):
                with tf.name_scope('weight'):
                    exec('W_%d = tf.Variable(tf.truncated_normal([self.layer[%d], self.layer[%d + 1]], stddev=0.1))' % (i, i, i))
                    exec('self.variable_summaries(var=W_%d)' % i)
                with tf.name_scope('bias'):
                    exec('b_%d = tf.Variable(tf.ones([self.layer[%d + 1]]) / 10)' % (i, i))
                    exec('self.variable_summaries(var=b_%d)' % i)
                with tf.name_scope('relu'):
                    exec('R_%d = tf.nn.relu(tf.matmul(X_%d, W_%d) + b_%d)' % (i, i, i, i))
                with tf.name_scope('drop'):
                    exec('X_%d = tf.nn.dropout(R_%d, self.keep_prob)' % (i + 1, i))

        with tf.name_scope('output'):
            with tf.name_scope('weight'):
                exec('W_%d = tf.Variable(tf.truncated_normal([self.layer[%d], self.layer[%d + 1]], stddev=0.1))' % (self.h_n, self.h_n, self.h_n))
                exec('self.variable_summaries(var=W_%d)' % self.h_n)
            with tf.name_scope('bias'):
                exec('b_%d = tf.Variable(tf.zeros([self.layer[%d + 1]]))' % (self.h_n, self.h_n))
                exec('self.variable_summaries(var=b_%d)' % self.h_n)
            with tf.name_scope('Y_logits'):
                self.Y_logits = eval('tf.matmul(X_%d, W_%d) + b_%d' % (self.h_n, self.h_n, self.h_n))
            with tf.name_scope('Y'):
                self.Y = tf.nn.softmax(self.Y_logits)

        with tf.name_scope('loss'):
            with tf.name_scope('cross_entropy'):
                self.cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=self.Y_logits, labels=self.Y_)
            with tf.name_scope('loss'):
                self.loss = tf.reduce_mean(self.cross_entropy)
                tf.summary.histogram('loss', self.loss)

        with tf.name_scope('accuracy'):
            with tf.name_scope('correct_prediction'):
                self.correct_prediction = tf.equal(tf.argmax(self.Y, 1), tf.argmax(self.Y_, 1))
            with tf.name_scope('accuracy'):
                self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))
                tf.summary.histogram('accuracy', self.accuracy)

        # with tf.name_scope('AdamOpt'):
        self.train_step = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

    @staticmethod
    def logs_clear():
        if 'logs' in os.listdir('./'):
            subprocess.Popen('rm -rf *', shell=True, cwd='./logs')
        else:
            subprocess.Popen('mkdir logs', shell=True)

    # noinspection SpellCheckingInspection
    @staticmethod
    def kill_tensorboard_process():
        for pid in psutil.pids():
            try:
                cmd = psutil.Process(pid=pid)
                if cmd.name() == 'tensorboard':
                    subprocess.call(['kill %d' % pid], shell=True)
            except psutil.NoSuchProcess:
                pass

    @staticmethod
    def variable_summaries(var):
        """Attach a lot of summaries to a Tensor (for Tensorboard visualization)."""
        with tf.name_scope('summaries'):
            mean = tf.reduce_mean(var)
            tf.summary.scalar('mean', mean)
            with tf.name_scope('stddev'):
                stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
            tf.summary.scalar('stddev', stddev)
            tf.summary.scalar('max', tf.reduce_max(var))
            tf.summary.scalar('min', tf.reduce_min(var))
            tf.summary.histogram('histogram', var)

    def train(self):
        
        init = tf.global_variables_initializer()
        sess = tf.Session()
        sess.run(init)
        
        self.logs_clear()
        merged = tf.summary.merge_all()
        writer = tf.summary.FileWriter(self.logs_dir, tf.get_default_graph())

        tf.set_random_seed(0)
        for i in range(500):
            # noinspection PyPep8Naming
            batch_X, batch_Y = self.mnist.train.next_batch(100)
            learning_rate = self.min_learning_rate + (self.max_learning_rate - self.min_learning_rate) * math.exp(-i / self.decay_speed)

            if (i + 1) % 20 == 0:
                a, c = sess.run([self.accuracy, self.loss], {self.X: batch_X, self.Y_: batch_Y, self.keep_prob: 1.0})
                print(str(i) + ": accuracy:" + str(a) + ",\tloss: " + str(c) + ",\t(learning rate:" + str(learning_rate) + ")")

                a, c = sess.run([self.accuracy, self.loss], {self.X: self.mnist.test.images, self.Y_: self.mnist.test.labels, self.keep_prob: 1.0})
                print(str(i) + ": test accuracy:" + str(a) + ",\ttest loss: " + str(c))

            summary, _ = sess.run([merged, self.train_step], {self.X: batch_X, self.Y_: batch_Y, self.keep_prob: 0.75, self.lr: learning_rate})
            writer.add_summary(summary, i)

        writer.close()

        self.kill_tensorboard_process()
        subprocess.Popen("tensorboard --logdir=%s" % self.logs_dir, shell=True)
        subprocess.Popen('firefox http://%s:%d/#graphs' % (socket.gethostname(), 6006), shell=True)

        for pid in psutil.pids():
            try:
                cmd = psutil.Process(pid=pid)
                if cmd.name() == 'tensorboard':
                    print('tensorboard pid: ', pid)
            except psutil.NoSuchProcess:
                pass


if __name__ == '__main__':
    obj = MnistDemo()
    obj.train()
