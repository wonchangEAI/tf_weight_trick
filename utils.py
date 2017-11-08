# utils for tf weight tricks
# mostly by Taesup Kim

import numpy as np
import tensorflow as tf

# Set value loading operations
def param_value_load_ops(param_tensor_list):
    """

    Args:
        param_tensor_list: list of trainable parameters

    Returns:
        load_ops_list: list of ops
        load_ph_list: list of placeholders
    """
    load_ops_list = []
    load_ph_list = []

    # For each parameter
    for param_tensor in param_tensor_list:
        # Make a copy of placeholder and load it
        param_load_ph = tf.placeholder(dtype=param_tensor.dtype,
                                       shape=param_tensor.shape)
        load_ph_list.append(param_load_ph)
        # Load ops with placeholder assigned to it
        load_ops_list.append(param_tensor.assign(param_load_ph))

    return load_ops_list, load_ph_list

# Get list of trainable parameters
param_tensor_list = tf.trainable_variables()

# TODO: figure out how to save and load tf params in numpy
# Load initial parameter list
# as list of ndarray
init_param_value_list = ...

# Load trained parameter list
# as list of ndarray
trnd_param_value_list = ...

# Build load operation
load_ops_list, load_ph_list = param_value_load_ops(param_tensor_list)

# CHECK: factoring possible?
# Open session
tf.global_variables_initializer().run()
with tf.Session() as sess:
    # linearly combine init and trained parameters
    # do looping for alpha
    for alpha in np.arange(0.0,1.01,.01):
        feed_dict = {}
        for load_ph, init_value, trnd_value in zip(load_ph_list, init_param_value_list, trnd_param_value_list):
            # linearly combine values
            feed_dict[load_ph] = init_value*(1.-alpha) + trnd_value*alpha

        sess.run(load_ops_list,
                 feed_dict=feed_dict)

        # do eval by using the graph
        cost_value = sess.run(cost,
                              feed_dict=...)


