import tensorflow as tf
import numpy as np


class MLP(object):
    """A simple MLP that has some configurability, mostly here to show how you can make a Keras
    like model in tensorflow. Since it helps with the 'Sessions are not python-friendly' stuff."""

    #
    # Below we have the part that involves creating the network
    #

    def __init__(self, input_size, nb_class):
        """Give certain parameters to the network"""

        #
        self.nb_class = nb_class

        # Create network
        tf.reset_default_graph()
        self._define_graph(input_size, nb_class)
        self._initialize_session()

    def _initialize_session(self):
        """Initialize session, variables, saver"""
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self._sess = tf.Session(config=config)
        self._sess.run(tf.global_variables_initializer())
        self._saver = tf.train.Saver()

    def _define_graph(self, input_size, nb_class):
        """Here we build the actual network, it can be made to be more customizable ..."""

        # Here we define inputs of the network
        self.img = tf.placeholder(dtype=tf.float32, shape=[None, input_size[0]], name='img')
        self.lbl = tf.placeholder(dtype=tf.float32, shape=[None, nb_class], name='lbl')
        self.trn = tf.placeholder(dtype=tf.bool, shape=[], name='trn')

        #
        # Yes, I know it can be done more low level or [name of fancy technique].
        # Do that yourself. The code below is mostly to show one of the many ways.
        #

        # Hidden layer
        self.ly1 = tf.layers.dense(self.img, 512, activation=tf.nn.relu, name='layer1')
        self.ly2 = tf.layers.dense(self.ly1, 512, activation=tf.nn.relu, name='layer2')

        # The output, a 'self made' loss function and the accuracy
        self.out = tf.layers.dense(self.ly2, nb_class, activation=tf.nn.softmax, name='out')
        self.lss = self._cross_entropy(self.out, self.lbl)

        correct_prediction = tf.equal(tf.argmax(self.out, axis=1), tf.argmax(self.lbl, axis=1))
        self.acc = tf.cast(correct_prediction, tf.float32)

        # Obtain an optimizer - 'something tries to minimize the loss by changing weights in a
        # smart way'. The basics involve gradient descent, google it ... There are many
        # variants, one of which is RMSProp, please experiment (If you ask me honestly,
        # I think vanilla gradient descent + momentum or decay is the best, but it takes (much)
        # longer).
        #
        # Using this optimizer we define what a trainig step is. Tensorflow takes care of the
        # backpropagation using this graph (google computational graphs and backpropagation
        # if you want to know more, or just read the deep learning book by Ian Goodfellow).
        optimizer = tf.train.RMSPropOptimizer(learning_rate=0.01)
        self.trainstep = optimizer.minimize(self.lss, name='train_step')

    @staticmethod
    def _cross_entropy(output, labels, eps=1e-8):
        """Calculate the cross entropy over the output"""
        return tf.reduce_sum(-labels * tf.log(output + eps), axis=1)

    #
    # Below we have the part that involves training the network
    #

    def fit(self, trn_data, val_data, nb_epoch, batch_sz, verbose):
        """Train the network, very basic you can make this as shiny as you want"""
        trn_loss_collection = []
        val_loss_collection = []
        trn_acc_collection = []
        val_acc_collection = []

        for epoch in range(nb_epoch):
            trn_loss, trn_acc = self._epoch(trn_data, batch_sz, True)
            val_loss, val_acc = self._epoch(val_data, batch_sz, False)
            if verbose == 1:
                print('Epoch={} - loss={}, acc={} - val_loss={}, val_acc={}'.format(
                    epoch,
                    np.round(trn_loss, 3),
                    np.round(trn_acc, 3),
                    np.round(val_loss, 3),
                    np.round(val_acc, 3)
                ))

            trn_loss_collection.append(trn_loss)
            val_loss_collection.append(val_loss)
            trn_acc_collection.append(trn_acc)
            val_acc_collection.append(val_acc)

        return trn_loss_collection, val_loss_collection, trn_acc_collection, val_acc_collection

    def _epoch(self, data, batch_sz, train):
        """A single pass over the data"""

        # Keeps track of the loss and accuracy in an ugly way
        total_lss = 0
        total_acc = 0

        # Extract the images and the labels
        img = data[0]
        lbl = data[1]

        # Shuffle
        idx = np.random.permutation(np.arange(img.shape[0]))
        idx_chunks = [idx[i:i + batch_sz] for i in range(0, len(idx), batch_sz)]

        for chunk in idx_chunks:
            # Get the feed_dict, what do we throw into the computational graph
            feed_dict = {
                self.img: img[chunk],
                self.lbl: lbl[chunk],
                self.trn: train
            }

            # Run it through the graph
            if train:
                _, batch_lss, batch_acc = self._sess.run(
                    fetches=[self.trainstep, self.lss, self.acc],
                    feed_dict=feed_dict
                )
            else:
                batch_lss, batch_acc = self._sess.run(
                    fetches=[self.lss, self.acc],
                    feed_dict=feed_dict
                )

            total_lss += np.sum(batch_lss)
            total_acc += np.sum(batch_acc)

        return 1 / img.shape[0] * total_lss, 1 / img.shape[0] * total_acc

    #
    # Below we have the part that involves using the network
    #

    def predict(self, data, batch_sz):
        """Predict the outcome for given data"""
        img = data[0]
        lbl = np.zeros()

        # Shuffle
        idx = np.arange(img.shape[0])
        idx_chunks = [idx[i:i + batch_sz] for i in range(0, len(idx), batch_sz)]

        for chunk in idx_chunks:
            # Get the feed_dict, what do we throw into the computational graph
            feed_dict = {
                self.img: img[chunk],
                self.lbl: lbl[chunk],
                self.trn: train
            }

            # Run it through the graph
            if train:
                _, batch_lss, batch_acc = self._sess.run(
                    fetches=[self.trainstep, self.lss, self.acc],
                    feed_dict=feed_dict
                )
            else:
                batch_lss, batch_acc = self._sess.run(
                    fetches=[self.lss, self.acc],
                    feed_dict=feed_dict
                )

            total_lss += np.sum(batch_lss)
            total_acc += np.sum(batch_acc)

        return 1 / img.shape[0] * total_lss, 1 / img.shape[0] * total_acc


    def load(self, path):
        pass

    def save(self, path):
        pass

# To test stuff
if __name__ == '__main__':
    inst_mlp = MLP(input_size=[784], nb_class=10)
