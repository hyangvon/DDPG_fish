# import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np

np.random.seed(1)
tf.random.set_random_seed(1)
# ####################  hyper parameters  ####################

LR_A = 0.0001   # learning rate for actor
LR_C = 0.0001   # learning rate for critic
GAMMA = 0.9     # reward discount
TAU = 0.8       # soft replacement
MEMORY_CAPACITY = 3000
BATCH_SIZE = 50 #原来是30


class DDPG(object):
    def __init__(self, a_dim, s_dim, a_bound,):
        self.memory = np.zeros((MEMORY_CAPACITY, s_dim * 2 + a_dim + 1), dtype=np.float32)
        self.pointer = 0
        self.memory_full = False
        self.sess = tf.Session()
        self.a_replace_counter, self.c_replace_counter = 0, 0

        self.a_dim, self.s_dim, self.a_bound = a_dim, s_dim, a_bound
        self.S = tf.placeholder(tf.float32, [None, s_dim], 's')
        self.S_ = tf.placeholder(tf.float32, [None, s_dim], 's_')
        self.R = tf.placeholder(tf.float32, [None, 1], 'r')

        with tf.variable_scope('Actor'):
            self.a = self._build_a(self.S, scope='eval', trainable=True)
            a_ = self._build_a(self.S_, scope='target', trainable=False)
        with tf.variable_scope('Critic'):
            # assign self.a = a in memory when calculating q for td_error,
            # otherwise the self.a is from Actor when updating Actor
            self.q = self._build_c(self.S, self.a, scope='eval', trainable=True)
            q_ = self._build_c(self.S_, a_, scope='target', trainable=False)

        # networks parameters
        self.ae_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/eval')
        self.at_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/target')
        self.ce_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/eval')
        self.ct_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/target')

        # target net replacement
        self.soft_replace = [[tf.assign(ta, (1 - TAU) * ta + TAU * ea), tf.assign(tc, (1 - TAU) * tc + TAU * ec)]
                             for ta, ea, tc, ec in zip(self.at_params, self.ae_params, self.ct_params, self.ce_params)]

        q_target = self.R + GAMMA * q_
        # in the feed_dic for the td_error, the self.a should change to actions in memory
        td_error = tf.losses.mean_squared_error(labels=q_target, predictions=self.q)
        self.ctrain = tf.train.AdamOptimizer(LR_C).minimize(td_error, var_list=self.ce_params)

        a_loss = - tf.reduce_mean(self.q)  # maximize the q
        self.atrain = tf.train.AdamOptimizer(LR_A).minimize(a_loss, var_list=self.ae_params)

        self.sess.run(tf.global_variables_initializer())

        self.var1 = 0.5
        self.var2 = 0.5

    def choose_action(self, s):
        action = self.sess.run(self.a, {self.S: s[None, :]})[0]
        action[0] = np.clip(np.random.normal(action[0], self.var1), 1, 21)
        action[1] = np.clip(np.random.normal(action[1], self.var2), 39, 61)
        return action
    
    def choose_action1(self, s):
        action = self.sess.run(self.a, {self.S: s[None, :]})[0]
        return action

    def learn(self):
        # soft target replacement
        self.sess.run(self.soft_replace)
        self.var1 = self.var1 * 0.999
        self.var2 = self.var2 * 0.999
        indices = np.random.choice(MEMORY_CAPACITY, size=BATCH_SIZE)
        bt = self.memory[indices, :]
        bs = bt[:, :self.s_dim]
        ba = bt[:, self.s_dim: self.s_dim + self.a_dim]
        br = bt[:, -self.s_dim - 1: -self.s_dim]
        bs_ = bt[:, -self.s_dim:]

        self.sess.run(self.atrain, {self.S: bs})
        self.sess.run(self.ctrain, {self.S: bs, self.a: ba, self.R: br, self.S_: bs_})

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, a, [r], s_))
        my_file = open('train.txt', 'a')
        store = ','.join(str(i) for i in transition)
        my_file.write(store + '\n')
        my_file.close()
        index = self.pointer % MEMORY_CAPACITY  # replace the old memory with new memory
        self.memory[index, :] = transition
        self.pointer += 1
        if self.pointer > MEMORY_CAPACITY:      # indicator for learning
            self.memory_full = True

    def _build_a(self, s, scope, trainable):
        with tf.variable_scope(scope):
            init_w = tf.keras.initializers.glorot_normal()
            init_b = tf.constant_initializer(0.001)
            net = tf.layers.dense(s, 300, activation=tf.nn.relu6, kernel_initializer=init_w,bias_initializer=init_b, name='l1', trainable=trainable)
            # net = tf.layers.dense(net, 200, activation=tf.nn.relu6, kernel_initializer=init_w,bias_initializer=init_b, name='l2', trainable=trainable)
            net = tf.layers.dense(net, 100, activation=tf.nn.relu, kernel_initializer=init_w, bias_initializer=init_b,name='l3', trainable=trainable)
            a = tf.layers.dense(net, self.a_dim, activation=tf.nn.tanh, kernel_initializer=init_w,bias_initializer=init_b, name='a', trainable=trainable)
            return tf.multiply(a, self.a_bound, name='scaled_a') + [10, 50]

    def _build_c(self, s, a, scope, trainable):
        with tf.variable_scope(scope):
            init_w = tf.keras.initializers.glorot_normal()
            init_b = tf.constant_initializer(0.001)
            n_l1 = 300
            w1_s = tf.get_variable('w1_s', [self.s_dim, n_l1], trainable=trainable)
            w1_a = tf.get_variable('w1_a', [self.a_dim, n_l1], trainable=trainable)
            b1 = tf.get_variable('b1', [1, n_l1], trainable=trainable)
            net = tf.nn.relu6(tf.matmul(s, w1_s) + tf.matmul(a, w1_a) + b1)
            # net = tf.layers.dense(net, 200, activation=tf.nn.relu6, kernel_initializer=init_w,bias_initializer=init_b, name='l4', trainable=trainable)
            net = tf.layers.dense(net, 100, activation=tf.nn.relu, kernel_initializer=init_w, bias_initializer=init_b, name='l5', trainable=trainable)
            return tf.layers.dense(net, 1, activation=tf.nn.tanh, kernel_initializer=init_w, bias_initializer=init_b, trainable=trainable)  # Q(s,a)

    def save(self):
        saver = tf.compat.v1.train.Saver()
        saver.save(self.sess, './params', write_meta_graph=False)
        print("Model saved.")

    def restore(self):
        saver = tf.compat.v1.train.Saver()
        saver.restore(self.sess, './params')
        print("Model restored.")

    def gotq(self, s, a):
        s = s[np.newaxis, :]
        a = a[np.newaxis, :]
        x = self.sess.run(self.q, feed_dict={self.S: s, self.a: a})
        return x
