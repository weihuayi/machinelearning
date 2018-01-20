import os
import math
import psutil
import socket
import subprocess

import tensorflow as tf
# noinspection SpellCheckingInspection
from tensorflow.examples.tutorials.mnist import input_data as mnist_data

tf.set_random_seed(0)


class MnistDemo:
    input_data_dir = './data/input_data'
    logs_dir = './logs'
    batch_size = 100
    conv_param = {0: {'height': 6, 'width': 6, 'in_channels': 1, 'out_channels': 24, 'stride': 1},
                  1: {'height': 5, 'width': 5, 'in_channels': 24, 'out_channels': 48, 'stride': 2},
                  2: {'height': 4, 'width': 4, 'in_channels': 48, 'out_channels': 64, 'stride': 2}}
    hidden_layer = [200]
    max_learning_rate = 0.02
    min_learning_rate = 0.0001
    decay_speed = 1600
    
    def __init__(self):
        # noinspection SpellCheckingInspection
        self.mnist = mnist_data.read_data_sets(self.input_data_dir, one_hot=True, reshape=False, validation_size=0)
        # noinspection PyPep8Naming
        temp_X, temp_Y = self.mnist.train.next_batch(self.batch_size)
        self.fig_size = [temp_X.shape[1], temp_X.shape[2]]
        # noinspection SpellCheckingInspection
        self.num_classifer = temp_Y.shape[1]
        self.cn = len(self.conv_param)
        self.hn = len(self.hidden_layer)
        
        if self.conv_param[0]['in_channels'] != 1:
            raise ValueError("the origin figure's channel is not equal to %d." % self.conv_param[0]['in_channels'])
        if self.cn != 1:
            for i in range(self.cn - 1):
                if self.conv_param[i]['out_channels'] != self.conv_param[i + 1]['in_channels']:
                    raise ValueError("convolutional layer_%d's out_channels is not equal to convolutional layer_%d's in_channels." % (i, i + 1))

        with tf.name_scope('input'):
            self.X = tf.placeholder(tf.float32, [None, self.fig_size[0], self.fig_size[1], 1], name='X')
            self.Y_ = tf.placeholder(tf.float32, [None, self.num_classifer], name='Y_')
            # noinspection PyPep8Naming,PyUnusedLocal
            X_0 = self.X
            
        with tf.name_scope('param'):
            self.lr = tf.placeholder(tf.float32, name='learning_rate')
            self.tst = tf.placeholder(tf.bool, name='tst')
            self.iter = tf.placeholder(tf.int32, name='iter')
            self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
            self.keep_prob_conv = tf.placeholder(tf.float32, name='keep_prob_conv')
            
        for i in range(len(self.conv_param)):
            with tf.name_scope('conv_%d' % i):
                with tf.name_scope('kernel'):
                    exec('K_%d = tf.Variable(tf.truncated_normal([%d, %d, %d, %d], stddev=0.1))' % (i, self.conv_param[i]['height'], self.conv_param[i]['width'], self.conv_param[i]['in_channels'], self.conv_param[i]['out_channels']))
                    exec('self.variable_summaries(var=K_%d)' % i)
                with tf.name_scope('bias'):
                    exec('B_%d = tf.Variable(tf.constant(0.1, tf.float32, [%d]))' % (i, self.conv_param[i]['out_channels']))
                    exec('self.variable_summaries(var=B_%d)' % i)
                with tf.name_scope('conv'):
                    exec('C_%d = tf.nn.conv2d(X_%d, K_%d, strides=[1, %d, %d, 1], padding="SAME")' % (i, i, i, self.conv_param[i]['stride'], self.conv_param[i]['stride']))
                with tf.name_scope('relu'):
                    with tf.name_scope('batch_norm'):
                        exec('N_%d, update_ema_%d = self.batchnorm(C_%d, self.tst, self.iter, B_%d, convolutional=True)' % (i, i, i, i))
                    with tf.name_scope('relu'):
                        exec('R_%d = tf.nn.relu(N_%d)' % (i, i))
                with tf.name_scope('drop'):
                    with tf.name_scope('noise'):
                        exec('noise = self.compatible_convolutional_noise_shape(R_%d)' % i)
                    with tf.name_scope('drop'):
                        exec('X_%d = tf.nn.dropout(R_%d, self.keep_prob_conv, noise)' % (i + 1, i))
        
        with tf.name_scope('full_conn_setting'):
            with tf.name_scope('full_connecting_layer_setting'):
                layer = eval("[X_%d.shape[1] * X_%d.shape[1] * self.conv_param[self.cn - 1]['out_channels']] + self.hidden_layer + [%d]" % (self.cn, self.cn, 10))  # TODO: 10
            with tf.name_scope('reshape_conv_output'):
                exec('X_%d = tf.reshape(X_%d, shape=[-1, %d])' % (self.cn, self.cn, layer[0]))

        for i in range(self.hn):
            with tf.name_scope('full_conn_layer_%d' % i):
                with tf.name_scope('weight'):
                    exec('W_%d = tf.Variable(tf.truncated_normal([%d, %d], stddev=0.1))' % (i, layer[i], layer[i + 1]))
                    exec('self.variable_summaries(var=W_%d)' % i)
                with tf.name_scope('bias'):
                    exec('b_%d = tf.Variable(tf.constant(0.1, tf.float32, [%d]))' % (i, layer[i + 1]))
                    exec('self.variable_summaries(var=b_%d)' % i)
                with tf.name_scope('relu'):
                    with tf.name_scope('batch_norm'):
                        exec('N_%d, update_ema_%d = self.batchnorm(tf.matmul(X_%d, W_%d), self.tst, self.iter, b_%d)' % (self.cn + i, self.cn + i, self.cn + i, i, i))
                    with tf.name_scope('relu'):
                        exec('R_%d = tf.nn.relu(N_%d)' % (self.cn + i, self.cn + i))
                with tf.name_scope('drop'):
                    exec('X_%d = tf.nn.dropout(R_%d, self.keep_prob)' % (self.cn + 1 + i, self.cn + i))

        with tf.name_scope('output'):
            with tf.name_scope('weight'):
                exec('W_%d = tf.Variable(tf.truncated_normal([%d, %d], stddev=0.1))' % (self.hn, layer[-2], layer[-1]))
                exec('self.variable_summaries(var=W_%d)' % self.hn)
            with tf.name_scope('bias'):
                exec('b_%d = tf.Variable(tf.constant(0.1, tf.float32, [%d]))' % (self.hn, layer[-1]))
                exec('self.variable_summaries(var=b_%d)' % self.hn)
            with tf.name_scope('Y_logits'):
                self.Y_logits = eval('tf.matmul(X_%d, W_%d) + b_%d' % (self.cn + self.hn, self.hn, self.hn))
            with tf.name_scope('Y'):
                self.Y = tf.nn.softmax(self.Y_logits)
        
        with tf.name_scope('update_ema'):
            self.update_ema = eval(('tf.group(update_ema_%d' + ', update_ema_%d'*(self.cn + self.hn - 1) + ')') % tuple(range(self.cn + self.hn)))
        
        with tf.name_scope('loss'):
            with tf.name_scope('cross_entropy'):
                cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=self.Y_logits, labels=self.Y_)
            with tf.name_scope('loss'):
                self.loss = tf.reduce_mean(cross_entropy)
                tf.summary.histogram('loss', self.loss)

        with tf.name_scope('accuracy'):
            with tf.name_scope('correct_prediction'):
                correct_prediction = tf.equal(tf.argmax(self.Y, 1), tf.argmax(self.Y_, 1))
            with tf.name_scope('accuracy'):
                self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
                tf.summary.histogram('accuracy', self.accuracy)
        
        self.train_step = tf.train.AdamOptimizer(self.lr).minimize(cross_entropy)

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

    # noinspection SpellCheckingInspection,PyPep8Naming
    @staticmethod
    def batchnorm(Y_logits, is_test, iteration, offset, convolutional=False):
        exp_moving_avg = tf.train.ExponentialMovingAverage(0.999, iteration)  # adding the iteration prevents from averaging across non-existing iterations
        # noinspection SpellCheckingInspection
        bnepsilon = 1e-5
        if convolutional:
            mean, variance = tf.nn.moments(Y_logits, [0, 1, 2])
        else:
            mean, variance = tf.nn.moments(Y_logits, [0])
        update_moving_everages = exp_moving_avg.apply([mean, variance])
        m = tf.cond(is_test, lambda: exp_moving_avg.average(mean), lambda: mean)
        v = tf.cond(is_test, lambda: exp_moving_avg.average(variance), lambda: variance)
        # noinspection PyPep8Naming
        Ybn = tf.nn.batch_normalization(Y_logits, m, v, offset, None, bnepsilon)
        return Ybn, update_moving_everages

    # noinspection SpellCheckingInspection
    @staticmethod
    def compatible_convolutional_noise_shape(var):
        noiseshape = tf.shape(var)
        noiseshape = noiseshape * tf.constant([1, 0, 0, 1]) + tf.constant([0, 1, 1, 0])
        return noiseshape

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

    def train(self):
        init = tf.global_variables_initializer()
        sess = tf.Session()
        sess.run(init)

        self.logs_clear()
        merged = tf.summary.merge_all()
        writer = tf.summary.FileWriter(self.logs_dir, tf.get_default_graph())

        tf.set_random_seed(0)

        for i in range(120):

            # noinspection PyPep8Naming
            batch_X, batch_Y = self.mnist.train.next_batch(self.batch_size)
            learning_rate = self.min_learning_rate + (self.max_learning_rate - self.min_learning_rate) * math.exp(-i / self.decay_speed)

            if (i + 1) % 20 == 0:
                a, c = sess.run([self.accuracy, self.loss], {self.X: batch_X, self.Y_: batch_Y, self.tst: False, self.keep_prob: 1.0, self.keep_prob_conv: 1.0})
                print(str(i) + ": accuracy:" + str(a) + ",\tloss: " + str(c) + ",\t(lr:" + str(learning_rate) + ")")

                a, c = sess.run([self.accuracy, self.loss], {self.X: self.mnist.test.images, self.Y_: self.mnist.test.labels, self.tst: True, self.keep_prob: 1.0, self.keep_prob_conv: 1.0})
                print(str(i) + ": test accuracy:" + str(a) + ",\ttest loss: " + str(c))

                # a, c = sess.run([self.accuracy, self.loss], {self.X: self.mnist.test.images, self.Y_: self.mnist.test.labels, self.tst: True, self.keep_prob: 1.0, self.keep_prob_conv: 1.0})
                # print(str(i) + ": test accuracy:" + str(a) + ",\ttest loss: " + str(c))

            summary, _ = sess.run([merged, self.train_step], {self.X: batch_X, self.Y_: batch_Y, self.lr: learning_rate, self.tst: False, self.keep_prob: 0.75, self.keep_prob_conv: 1.0})
            sess.run(self.update_ema, {self.X: batch_X, self.Y_: batch_Y, self.tst: False, self.iter: i, self.keep_prob: 1.0, self.keep_prob_conv: 1.0})
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
