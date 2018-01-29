import tensorflow as tf


def relu(x, name='relu6'):
    return tf.nn.relu6(x, name)


def batch_norm(x, momentum=0.9, epsilon=1e-5, train=True, name='batch_norm'):
    return tf.contrib.layers.batch_norm(x,
                      decay=momentum,
                      updates_collections=None,
                      epsilon=epsilon,
                      scale=True,
                      is_training=train,
                      scope=name)


def conv2d(input_, output_dim, k_h=3, k_w=3, d_h=1, d_w=1, stddev=0.02,name='conv2d'):
  with tf.variable_scope(name):
    w = tf.get_variable('w', [k_h, k_w, input_.get_shape()[-1], output_dim],
              initializer=tf.truncated_normal_initializer(stddev=stddev))
    conv = tf.nn.conv2d(input_, w, strides=[1, d_h, d_w, 1], padding='SAME')

    biases = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.0))
    conv = tf.nn.bias_add(conv, biases)

    return conv


def conv2d_block(input, out_dim, k, s, name, is_train):
    with tf.name_scope(name):
        net = conv2d(input, out_dim, k, k, s, s, name=name)
        net = batch_norm(net, train=is_train)
        net = relu(net)
        return net


def conv_1x1(input, output_dim, name='conv1x1'):
    with tf.name_scope(name):
        return conv2d(input, output_dim, k_h=1, k_w=1, d_h=1, d_w=1, stddev=0.02, name=name)


def dwise_conv(input, k_h=3, k_w=3, channel_multiplier= 1, strides=[1,1,1,1], padding='SAME', stddev=0.02, name='dwise_conv'):
    with tf.name_scope(name):
        with tf.variable_scope(name):
            in_channel=input.get_shape().as_list()[-1]
            w = tf.get_variable('w', [k_h, k_w, in_channel, channel_multiplier],
                                initializer=tf.truncated_normal_initializer(stddev=stddev))
            return tf.nn.depthwise_conv2d(input, w, strides, padding, rate=None,name=None,data_format=None)


def res_block(input, expansion_ratio, out_put_dim, stride, name, is_train, hyperlink=True):
    with tf.name_scope(name), tf.variable_scope(name):
        # pw
        bottleneck_dim=round(expansion_ratio*input.get_shape().as_list()[-1])
        net = conv_1x1(input, bottleneck_dim, name='pw')
        net = batch_norm(net, train=is_train, name='pw')
        net = relu(net)
        # dw
        net = dwise_conv(net, strides=[1, stride, stride, 1], name='dw')
        net = batch_norm(net, train=is_train, name='dw')
        net = relu(net)
        # pw & linear
        net = conv_1x1(net, out_put_dim, name='pw_linear')
        net = batch_norm(net, train=is_train, name='pw_linear')

        # element wise add, only for stride==1
        if hyperlink and stride == 1:
            in_dim=int(input.get_shape().as_list()[-1])
            if in_dim != out_put_dim:
                ins=conv_1x1(input=input, output_dim=out_put_dim, name='ex_dim')
                net=ins+net
            else:
                net=input+net

        return net


def global_avg(x):
    net=tf.layers.average_pooling2d(x, x.get_shape()[1:-1], 1)
    return net


def flatten(x):
    #flattened=tf.reshape(input,[x.get_shape().as_list()[0], -1])  # or, tf.layers.flatten(x)
    return tf.contrib.layers.flatten(x)