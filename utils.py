# utils for tf weight tricks
# mostly by Taesup Kim

# Set value loading operations
def param_value_load_ops(param_tensor_list):
    load_ops_list = []
    load_ph_list = []

    # For each parameter
    for param_tensor in param_tensor_list:
        # Set value loading operation
        param_load_ph = tf.placeholder(dtype=param_tensor.dtype,
                                       shape=param_tensor.shape)
        load_ph_list.append(param_load_ph)
        load_ops_list.append(param_tensor.assign(param_load_ph))

    # Return list of operations
    return load_ops_list, load_ph_list

# TODO: make a graph
# TODO: save_init_param, train, save_final_param

# Get list of trainable parameters
# use tf.trainable_variables()
# TODO: look for saving tf params in numpy
param_tensor_list = ...

# Load initial parameter list
init_param_value_list = ...

# Load final parameter list
final_param_value_list = ...

# Build load operation
load_ops_list, load_ph_list = param_value_load_ops(param_tensor_list)

# Open session
tf.global_variables_initializer().run()
with tf.Session() as sess:
    # linear combine init and final parameter

    # do looping for alpha
    for alpha in [0.0, 0.1,.., 1.0]:
	    feed_dict = {}
	    for load_ph, init_value, final_value in zip(load_ph_list, init_param_value_list, final_param_value_list):
	        # combine values
	        feed_dict[load_ph] = init_value*(1.-alpha) + final_value*alpha

	    sess.run(load_ops_list,
	             feed_dict=feed_dict)

	    # do eval by using the graph
	    


