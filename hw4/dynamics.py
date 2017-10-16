import tensorflow as tf
import numpy as np
import consts

EPSILON = 1e-18

# Predefined function to build a feedforward neural network
def build_mlp(input_placeholder, 
              output_size,
              scope, 
              n_layers=2, 
              size=500, 
              activation=tf.tanh,
              output_activation=None
              ):
    out = input_placeholder
    with tf.variable_scope(scope):
        for _ in range(n_layers):
            out = tf.layers.dense(out, size, activation=activation)
        out = tf.layers.dense(out, output_size, activation=output_activation)
    return out

class NNDynamicsModel():
    def __init__(self, 
                 env, 
                 n_layers,
                 size, 
                 activation, 
                 output_activation, 
                 normalization,
                 batch_size,
                 iterations,
                 learning_rate,
                 sess
                 ):
        """ YOUR CODE HERE """
        """ Note: Be careful about normalization """

        # Initialize constants for training
        self.mean_states = normalization[consts.NORMALIZATION_KEY_MEAN_STATES]
        self.std_states = normalization[consts.NORMALIZATION_KEY_STD_STATES]
        self.mean_deltas = normalization[consts.NORMALIZATION_KEY_MEAN_STATES]
        self.std_deltas = normalization[consts.NORMALIZATION_KEY_STD_DELTAS]
        self.mean_actions = normalization[consts.NORMALIZATION_KEY_MEAN_ACTIONS]
        self.std_actions = normalization[consts.NORMALIZATION_KEY_STD_ACTIONS]

        self.batch_size = batch_size
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.sess = sess

        # Create Tensorflow graph

        self.action_shape = self.env.action_space.shape()
        self.state_shape = self.env.state_space.shape()

        t_action_shape = tuple([None] + list(self.action_shape))
        t_state_shape = tuple([None] + list(self.state_shape))

        self.t_actions = tf.placeholder(tf.float32, t_action_shape) 
        self.t_states = tf.placeholder(tf.float32, t_state_shape)
        self.t_delta_labels = tf.placeholder(tf.float32, t_state_shape)

        t_mlp_inputs = tf.concat([self.t_actions, self.t_states], axis=0)
        self.t_state_deltas = build_mlp(input_placeholder=t_mlp_inputs,
                                        output_size=self.state_shape,
                                        scope=None,
                                        n_layers=n_layers,
                                        size=size,
                                        activation=activation,
                                        output_activation=output_activation)

        self.t_loss = tf.nn.l2_loss(self.t_state_deltas - self.t_delta_labels)
        self.t_train = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.t_loss)

        # Model variables will be initialized prior to model use in `main.py`

    def fit(self, data):
        """
        Write a function to take in a dataset of (unnormalized)states, (unnormalized)actions, (unnormalized)next_states and fit the dynamics model going from normalized states, normalized actions to normalized state differences (s_t+1 - s_t)
        """
        """YOUR CODE HERE """

        states = np.array([state for state,_,_ in data], dtype=np.float32)
        actions = np.array([action for _,action,_ in data], dtype=np.float32)
        next_states = np.array([next_state for _,_,next_state in data], dtype=np.float32)

        norm_states = (states - self.mean_states) / (self.std_states + EPSILON)
        norm_actions = (actions - self.mean_actions) / (self.std_actions + EPSILON)

        state_deltas = (next_states - states)

        # TODO: CHECK THIS!!!
        norm_state_deltas = (state_deltas - self.mean_deltas) / (self.std_deltas + EPSILON)

        for _ in range(self.iterations):
            batch_idx = 0
            while batch_idx + self.batch_size < len(data):
                states_batch = norm_states[batch_idx : batch_idx + self.batch_size]
                actions_batch = norm_actions[batch_idx : batch_idx + self.batch_size]
                deltas_batch = norm_state_deltas[batch_idx : batch_idx + self.batch_size]

                feed_dict = {
                    self.t_states : states_batch,
                    self.t_actions : actions_batch,
                    self.t_delta_labels : deltas_batch
                }

                loss, _ = self.sess.run([self.t_loss, self.t_train], feed_dict=feed_dict)
                print("LOSS: {}".format(loss))

                batch_idx += self.batch_size

    def predict(self, states, actions):
        """ Write a function to take in a batch of (unnormalized) states and (unnormalized) actions and return the (unnormalized) next states as predicted by using the model """
        """ YOUR CODE HERE """

        norm_states = (states - self.mean_states) / (self.std_states + EPSILON)
        norm_actions = (actions - self.mean_actions) / (self.std_actions + EPSILON)

        feed_dict = {
            self.t_states : norm_states,
            self.norm_actions : norm_actions
        }
        norm_deltas = self.sess.run(self.t_state_deltas, feed_dict=feed_dict)
        final_deltas = (self.std_deltas * norm_deltas) + self.mean_deltas

        return final_deltas