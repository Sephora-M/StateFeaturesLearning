import tensorflow as tf
from numpy import nonzero
from numpy import random
from numpy import array
import math

from tensorflow.losses import Reduction


class DiscountedNode2Vec:

    def __init__(self, num_states, num_actions, dimension, window_size, walks, actions, discount):
        self.N = num_states
        self.num_actions = num_actions
        self.dimension = dimension
        self.window_size = window_size
        self.walks = walks
        self.actions = actions
        self.gamma = discount

    def encode_one_hot(self, i, value=1):

        if i < 0 or i >= self.N:
            raise ValueError('the interger i to be encoded must be in range [0, %d) ' % self.N)

        onehot = [0] * self.N
        onehot[i] = value

        return onehot

    def get_neighborhood(self, walk, onehot_output):

        input = []
        output = []
        dictionary = {}
        for current_node_index in range(len(walk)):
            dictionary[walk[current_node_index]] = self.encode_one_hot(walk[current_node_index])
            for w in range(1, self.window_size +1): #range(-self.window_size, self.window_size + 1):
                if 0 <= current_node_index + w < len(walk) and w != 0:
                    # input.append(self.encode_one_hot(walk[current_node_index]))
                    input.append(walk[current_node_index])
                    if onehot_output:
                        output.append(self.encode_one_hot(walk[current_node_index + w], self.gamma ** (abs(w)-1)))
                    else:
                        output.append(walk[current_node_index + w])

        return input, output, dictionary

    def get_neighborhood_with_actions(self, walk, actions, onehot_output):

        input = []
        output = []
        dictionary = {}
        for current_node_index in range(len(walk)):
            dictionary[walk[current_node_index]] = self.encode_one_hot(walk[current_node_index])
            for w in range(1, self.window_size +1):
                if 0 <= current_node_index + w < len(walk) and w != 0:
                    # input.append(self.encode_one_hot(walk[current_node_index]))
                    input.append([walk[current_node_index],actions[current_node_index]])
                    if onehot_output:
                        output.append(self.encode_one_hot(walk[current_node_index + w], self.gamma ** (abs(w)-1)))
                    else:
                        output.append([walk[current_node_index + w],actions[current_node_index+w]])

        return input, output, dictionary

    def process_walks(self, onehot_output=True):

        input_data = []
        y_true = []
        dictionary = {}

        for walk in self.walks:
            input, output, vocab = self.get_neighborhood(walk, onehot_output)
            input_data.extend(input)
            y_true.extend(output)
            dictionary.update(vocab)

        return input_data, y_true, dictionary

    def process_walks_with_actions(self, onehot_output=True):

        input_data = []
        y_true = []
        dictionary = {}

        for i in range(len(self.walks)):
            input, output, vocab = self.get_neighborhood_with_actions(self.walks[i], self.actions[i], onehot_output)
            input_data.extend(input)
            y_true.extend(output)
            dictionary.update(vocab)

        return input_data, y_true, dictionary

    @staticmethod
    def next_batch(input_data, y_true, batch_size, i):
        start = batch_size * (i - 1)
        end = batch_size * (i - 1) + batch_size
        if end >= len(input_data):
            end = -1
        batch_x = input_data[start: end]
        batch_y = y_true[start: end]
        return batch_x, batch_y

    def discounted_skipgram(self):

        input_node = tf.placeholder("float", [None, self.N])
        output_node = tf.placeholder("float", [None, self.N])

        weights = {'W1': tf.Variable(tf.random_normal([self.N, self.dimension])),
                   'W2': tf.Variable(tf.random_normal([self.dimension, self.N]))}

        biases = {'b1': tf.Variable(tf.random_normal([self.dimension])), 'b2': tf.Variable(tf.random_normal([self.N]))}

        embedding_layer = tf.add(tf.matmul(input_node, weights['W1']), biases['b1'])
        output = tf.nn.sigmoid(tf.add(tf.matmul(embedding_layer, weights['W2']), biases['b2']))

        return input_node, output_node, embedding_layer, output

    def train_discounted_n2v(self, batch_size=32, learning_rate=0.01, num_epochs=2):

        # input_node, output_node, embedding_layer, y_pred = self.discounted_skipgram()

        input_node = tf.placeholder(tf.int32, shape=[None])
        output_node = tf.placeholder(tf.float32, shape=[None,self.N])

        embedding_layer = tf.Variable(tf.random_uniform([self.N, self.dimension], -1.0, 1.0))
        embed = tf.nn.embedding_lookup(embedding_layer, input_node)

        initializer = tf.contrib.layers.xavier_initializer()
        weights = tf.Variable(initializer([self.N, self.dimension]))
        biases = tf.Variable(tf.zeros([self.N]))
        # hidden_out = tf.nn.sigmoid(tf.matmul(embed, tf.transpose(weights)) + biases)
        hidden_out = tf.matmul(embed, tf.transpose(weights)) + biases

        # convert train_context to a one-hot format
        # train_one_hot = tf.one_hot(output_node, self.N)
        # loss = tf.reduce_mean(tf.losses.mean_squared_error(labels=output_node, predictions=hidden_out, reduction=Reduction.SUM_OVER_BATCH_SIZE))
        loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=hidden_out,
                                                                               labels=output_node))

        # Construct the SGD optimizer using a learning rate of 1.0.
        # optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
        optimizer = tf.train.AdamOptimizer().minimize(loss)

        input_data, y_true, dictionary = self.process_walks()

        num_steps = int(len(input_data) / batch_size)

        # Start Training

        config = tf.ConfigProto(
            device_count={'GPU': 0}
        )
        # saver = tf.train.Saver(tf.global_variables(), max_to_keep=1)

        # Start a new TF session
        with tf.Session(config=config) as sess:
            with tf.variable_scope("model") as scope:

                sess.run(tf.global_variables_initializer())
                scope.reuse_variables()

                result_dic = {}
                all_batch_losses = []
                all_epoch_losses = []
                # Training
                i = 0
                for i in range(1, num_epochs + 1):
                    # shuffle data
                    # random.shuffle(input_data)
                    for j in range(1, num_steps + 1):
                        # Prepare Data
                        # Get the next batch of data
                        batch_x, batch_y = self.next_batch(input_data, y_true, batch_size, j)
                        # print(batch_y)

                        # Run optimization op (backprop) and cost op (to get loss value)
                        _, batch_loss = sess.run([optimizer, loss], feed_dict={input_node: batch_x, output_node: batch_y})

                        all_batch_losses.append(batch_loss)
                        # Display logs per step and save current model

                        # if j % 10 == 0 or j == 1:
                        #     print('Epoch %i, Step %i: Minibatch Loss: %f  \n' % (i, j, batch_loss))

                    epoch_loss = sess.run(loss, feed_dict={input_node: input_data, output_node: y_true})
                    print('Epoch %i Loss: %f' % (i, epoch_loss))
                    # print(embedding_layer.eval())
                    # all_epoch_losses.append(epoch_loss)

                result_dic["epoch_losses"] = all_epoch_losses
                result_dic["batch_losses"] = all_batch_losses

                embeddings = {}

                # for node in range(self.N):
                #     embeddings[str(node)] = array([0] * self.dimension)
                # if i > 0:
                #     epoch_loss = sess.run(loss, feed_dict={input_node: input_data, output_node: y_true})
                #     print('Training finished, epoch %i Loss: %f  \n' % (i, epoch_loss))

                for node in dictionary.keys():
                    embeddings[str(node)] = sess.run(embed, feed_dict={input_node: [node]})[0]

        return embeddings, result_dic

    def train_s2v(self, batch_size=32, learning_rate=0.01, num_epochs=2):

        # input_node, output_node, embedding_layer, y_pred = self.discounted_skipgram()

        input_state_action = tf.placeholder(tf.int32, shape=[None, 2])
        output_node = tf.placeholder(tf.float32, shape=[None, self.N])

        embedding_layer = tf.Variable(tf.random_uniform([self.N, self.num_actions, self.dimension], -1.0, 1.0))
        embed = tf.gather_nd(embedding_layer, input_state_action)

        initializer = tf.contrib.layers.xavier_initializer()
        weights = tf.Variable(initializer([self.N, self.dimension]))
        biases = tf.Variable(tf.zeros([self.N]))
        # hidden_out = tf.nn.sigmoid(tf.matmul(embed, tf.transpose(weights)) + biases)
        hidden_out = tf.matmul(embed, tf.transpose(weights)) + biases

        # convert train_context to a one-hot format
        # train_one_hot = tf.one_hot(output_node, self.N)
        # loss = tf.reduce_mean(tf.losses.mean_squared_error(labels=output_node, predictions=hidden_out, reduction=Reduction.SUM_OVER_BATCH_SIZE))
        loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=hidden_out,
                                                                               labels=output_node))

        # Construct the SGD optimizer using a learning rate of 1.0.
        # optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
        optimizer = tf.train.AdamOptimizer().minimize(loss)

        input_data, y_true, dictionary = self.process_walks_with_actions()

        num_steps = int(len(input_data) / batch_size)

        # Start Training

        config = tf.ConfigProto(
            device_count={'GPU': 0}
        )
        # saver = tf.train.Saver(tf.global_variables(), max_to_keep=1)

        # Start a new TF session
        with tf.Session(config=config) as sess:
            with tf.variable_scope("model") as scope:

                sess.run(tf.global_variables_initializer())
                scope.reuse_variables()

                result_dic = {}
                all_batch_losses = []
                all_epoch_losses = []
                # Training
                i = 0
                for i in range(1, num_epochs + 1):
                    # shuffle data
                    # random.shuffle(input_data)
                    for j in range(1, num_steps + 1):
                        # Prepare Data
                        # Get the next batch of data
                        batch_x, batch_y = self.next_batch(input_data, y_true, batch_size, j)
                        # print(batch_y)

                        # Run optimization op (backprop) and cost op (to get loss value)
                        _, batch_loss = sess.run([optimizer, loss], feed_dict={input_state_action: batch_x, output_node: batch_y})

                        all_batch_losses.append(batch_loss)
                        # Display logs per step and save current model

                        # if j % 10 == 0 or j == 1:
                        #     print('Epoch %i, Step %i: Minibatch Loss: %f  \n' % (i, j, batch_loss))

                    epoch_loss = sess.run(loss, feed_dict={input_state_action: input_data, output_node: y_true})
                    print('Epoch %i Loss: %f' % (i, epoch_loss))
                    # print(embedding_layer.eval())
                    # all_epoch_losses.append(epoch_loss)

                result_dic["epoch_losses"] = all_epoch_losses
                result_dic["batch_losses"] = all_batch_losses

                embeddings = {}

                # for node in range(self.N):
                #     embeddings[str(node)] = array([0] * self.dimension)
                # if i > 0:
                #     epoch_loss = sess.run(loss, feed_dict={input_node: input_data, output_node: y_true})
                #     print('Training finished, epoch %i Loss: %f  \n' % (i, epoch_loss))

                for node in dictionary.keys():
                    for action in range(self.num_actions):
                        embeddings[node, action] = sess.run(embed, feed_dict={input_state_action: [[node, action]]})[0]

        return embeddings, result_dic

    def train(self, batch_size=32, learning_rate=0.01, num_epochs=2):

        # input_node, output_node, embedding_layer, y_pred = self.discounted_skipgram()

        input_node = tf.placeholder(tf.int32, shape=[None])
        output_node = tf.placeholder(tf.int32, shape=[None,1])

        embedding_layer = tf.Variable(tf.random_uniform([self.N, self.dimension], -1.0, 1.0))
        embed = tf.nn.embedding_lookup(embedding_layer, input_node)

        initializer = tf.contrib.layers.xavier_initializer()
        weights = tf.Variable(initializer([self.N, self.dimension]))
        biases = tf.Variable(tf.zeros([self.N]))
        hidden_out = tf.matmul(embed, tf.transpose(weights)) + biases

        # convert train_context to a one-hot format
        train_one_hot = tf.one_hot(output_node, self.N)
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=hidden_out,
                                                                               labels=train_one_hot))
        # Construct the SGD optimizer using a learning rate of 1.0.
        # optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy)
        optimizer = tf.train.AdamOptimizer().minimize(cross_entropy)

        input_data, y_true, dictionary = self.process_walks(False)

        num_steps = int(len(input_data) / batch_size)

        # Start Training

        config = tf.ConfigProto(
            device_count={'GPU': 0}
        )
        # saver = tf.train.Saver(tf.global_variables(), max_to_keep=1)

        # Start a new TF session
        with tf.Session(config=config) as sess:
            with tf.variable_scope("model") as scope:

                sess.run(tf.global_variables_initializer())
                scope.reuse_variables()

                result_dic = {}
                all_batch_losses = []
                all_epoch_losses = []
                # Training
                for i in range(1, num_epochs + 1):
                    # shuffle data
                    # random.shuffle(input_data)
                    for j in range(1, num_steps + 1):
                        # Prepare Data
                        # Get the next batch of data
                        batch_x, batch_y = self.next_batch(input_data, y_true, batch_size, j)
                        # print(batch_y)

                        # Run optimization op (backprop) and cost op (to get loss value)
                        _, batch_loss = sess.run([optimizer, cross_entropy], feed_dict={input_node: batch_x, output_node: array([batch_y]).T})

                        all_batch_losses.append(batch_loss)
                        # Display logs per step and save current model

                        # if j % 10 == 0 or j == 1:
                        #     print('Epoch %i, Step %i: Minibatch Loss: %f  \n' % (i, j, batch_loss))

                    epoch_loss = sess.run(cross_entropy, feed_dict={input_node: input_data, output_node: array([y_true]).T})
                    print('Epoch %i Loss: %f  \n' % (i, epoch_loss))
                    # print(embedding_layer.eval())
                    all_epoch_losses.append(epoch_loss)

                result_dic["epoch_losses"] = all_epoch_losses
                result_dic["batch_losses"] = all_batch_losses

                embeddings = {}

                # for node in range(self.N):
                #     embeddings[str(node)] = array([0] * self.dimension)

                for node in dictionary.keys():
                    print(node)
                    embeddings[str(node)] = sess.run(embed, feed_dict={input_node: [node]})[0]

        return embeddings, result_dic

    def train2(self, batch_size=32, learning_rate=0.01, num_epochs=10):

        input_node, output_node, embedding_layer, y_pred = self.discounted_skipgram()

        input_data, y_true, dictionary = self.process_walks()

        num_steps = int(len(input_data) / batch_size)

        # Define loss and optimizer, minimize the squared error
        loss = tf.reduce_mean(tf.pow(output_node - y_pred, 2))
        # optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(loss)
        optimizer = tf.train.AdamOptimizer().minimize(loss)

        # Start Training

        config = tf.ConfigProto(
            device_count={'GPU': 0}
        )
        # saver = tf.train.Saver(tf.global_variables(), max_to_keep=1)

        # Start a new TF session
        with tf.Session(config=config) as sess:
            with tf.variable_scope("model") as scope:

                sess.run(tf.global_variables_initializer())
                scope.reuse_variables()

                result_dic = {}
                all_batch_losses = []
                all_epoch_losses = []
                # Training
                for i in range(1, num_epochs + 1):
                    # shuffle data
                    # random.shuffle(input_data)
                    for j in range(1, num_steps + 1):
                        # Prepare Data
                        # Get the next batch of data
                        batch_x, batch_y = self.next_batch(input_data, y_true, batch_size, j)

                        # Run optimization op (backprop) and cost op (to get loss value)
                        _, batch_loss = sess.run([optimizer, loss], feed_dict={input_node: batch_x, output_node: batch_y})
                        all_batch_losses.append(batch_loss)
                        # Display logs per step and save current model

                        if j % 10 == 0 or j == 1:
                            print('Epoch %i, Step %i: Minibatch Loss: %f  \n' % (i, j, batch_loss))

                    epoch_loss = sess.run(loss, feed_dict={input_node: input_data, output_node: y_true})
                    all_epoch_losses.append(epoch_loss)

                result_dic["epoch_losses"] = all_epoch_losses
                result_dic["batch_losses"] = all_batch_losses

                embeddings = {}

                # for node in range(self.N):
                #     embeddings[str(node)] = array([0] * self.dimension)

                for node in dictionary.keys():
                    embeddings[str(node)] = sess.run(embedding_layer, feed_dict={input_node:  [dictionary[node]]})[0]

        return embeddings, result_dic