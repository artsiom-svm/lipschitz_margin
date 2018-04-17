import tensorflow as tf
import numpy as np

class SimpleANN:

    def __init__(self, n_dim, y_dim, learning_rate=1e-4, margin_rate=0.0):
        '''
        Creates graph for 3 fully connected layers with softplus activation
        at hidden layers and sigmoid in output layer.
        
        Args:   
            n_dim: Dimention size of input vector
            y_dim: Dimention size of output vector
            learning_rate: The learning rate for the SGD
            margin_rate: regularization constant for lipschitz margin
        '''

        self.session = tf.Session()

        self._n_dim = n_dim
        self._y_dim = y_dim

        self.x_placehold = tf.placeholder(shape=(None, n_dim), dtype=tf.float32, name="Input")
        self.y_placehold = tf.placeholder(shape=(None, y_dim), dtype=tf.float32, name="Labels")

        self.y = self._layers(n_dim, y_dim)
        
        self.loss_tensor = self._loss(self.y, self.y_placehold, margin_rate);
        self.update_op = self._optimizer(self.loss_tensor, learning_rate)

        self.session.run(tf.global_variables_initializer())


    def _cross_entropy_loss(self, y, y_gt, clip_min=1e-6):
        '''
        Simple softmax cross entropy loss
        '''
        # clip number near zero to avoid underflow 
        # max can be 1.0 since we are using sigmoid activation
        _y = tf.clip_by_value(y, clip_min, 1.0)
        cross_entropy_loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_gt, logits=_y, name="cross_entropy_loss")
        return cross_entropy_loss 

    def _margin_loss(self, y, labels):
        '''
        Calculates lipschitz margin loss by:
        loss = n^-1 \sum_i max(y_j - y_i)
        '''
        # mean or sum?
        # TODO: check math
        margin_loss = tf.reduce_mean(tf.reduce_max(labels - y, axis=1), name="margin_loss")
        return margin_loss

    def _regularization_loss(self):
        '''
        Simple L2 loss calculations
        '''
        with tf.name_scope("L2 loss") as scope:
            l2_loss = tf.reduce_all([p**2 for p in self._params], name="l2")
        return l2_loss

    def _loss(self, y, y_gt, margin_rate, regularization_rate):
        '''
        Loss is considered to be :
        cross entropy + regularization_rate * L2 + margin_rate * lipschitz_margin
        '''
        with tf.name_scope("Losses") as scope:
            labels = self._eval(y)
            total_loss = tf.add(self._cross_entropy_loss(y, y_gt), 
            margin_rate * self._margin_loss(y, labels) +
            regularization_rate * self._regularization_loss(), name="total_loss")
        return total_loss

    def _optimizer(self, loss, learning_rate):
        '''
        For optimization was used default adam optimizer
        '''
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss,name="Optimizer")
        return optimizer

    def _layers(self, n_dim, y_dim):
        '''
        Creates graphs for layers, here is used next simple model:
              
         n_di        512             1024           128         y_dim
        input --> (softplus) -->  (sotfplus) --> (softplus) -> (sigmoid) -> output
        '''
        # inintialize with random weights
        W1_init = np.random.randn(n_dim, 512) / (n_dim + 512)
        b1_init = np.random.randn(512) / 512
        W2_init = np.random.randn(512, 1024) / (512 + 1024)
        b2_init = np.random.randn(1024) / 1024
        W3_init = np.random.randn(1024, 128) / (1024  +128)
        b3_init = np.random.randn(128) / 128
        W4_init = np.random.randn(128, y_dim) / (128 + y_dim)
        b4_init = np.random.randn(y_dim) / y_dim
        

        # create variables for graph
        W1 = tf.Variable(initial_value=W1_init, dtype=tf.float32, name="W1")
        W2 = tf.Variable(initial_value=W2_init, dtype=tf.float32, name="W2")
        W3 = tf.Variable(initial_value=W3_init, dtype=tf.float32, name="W3")
        W4 = tf.Variable(initial_value=W4_init, dtype=tf.float32, name="W4")
        b1 = tf.Variable(initial_value=b1_init, dtype=tf.float32, name="b1")
        b2 = tf.Variable(initial_value=b2_init, dtype=tf.float32, name="b2")
        b3 = tf.Variable(initial_value=b3_init, dtype=tf.float32, name="b3")
        b4 = tf.Variable(initial_value=b4_init, dtype=tf.float32, name="b4")


        # combine all params together
        self._params = [W1, W2, W3, W4, b1, b2, b3, b4]

        # create graph model
        with tf.name_scope("Model") as scope:
            h1 = tf.nn.softplus(tf.matmul(self.x_placehold, W1) + b1, name="layer1")
            h2 = tf.nn.softplus(tf.matmul(h1, W2) + b2, name="layer2")
            h3 = tf.nn.softplus(tf.matmul(h2, W3) + b3, name="layer3")
            out = tf.nn.sigmoid(tf.matmul(h3, W4) + b4, name="output_layer")
        return out

    def _eval(self, y):
        '''
        Evaluate y to hot vector with 1 at argmax(y)
        '''
        pass
    