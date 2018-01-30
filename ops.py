import tensorflow as tf


def relu(x, name='relu6'):
    return tf.nn.relu6(x, name)


def batch_norm(x, momentum=0.9, epsilon=1e-5, train=True, name='bn'):
    return tf.contrib.layers.batch_norm(x,
                      decay=momentum,
                      updates_collections=None,
                      epsilon=epsilon,
                      scale=True,
                      is_training=train,
                      scope=name)


def conv2d(input_, output_dim, k_h, k_w, d_h, d_w, weight_decay, stddev=0.02, name='conv2d', bias=True):
  with tf.variable_scope(name):
    w = tf.get_variable('w', [k_h, k_w, input_.get_shape()[-1], output_dim],
              initializer=tf.truncated_normal_initializer(stddev=stddev),
                  regularizer=tf.contrib.layers.l2_regularizer(weight_decay))
    conv = tf.nn.conv2d(input_, w, strides=[1, d_h, d_w, 1], padding='SAME')
    if bias:
        biases = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.0))
        conv = tf.nn.bias_add(conv, biases)

    return conv


def conv2d_block(input, out_dim, k, s, weight_decay, is_train, name):
    with tf.name_scope(name):
        net = conv2d(input, out_dim, k, k, s, s, weight_decay=weight_decay,name=name)
        net = batch_norm(net, train=is_train, name=name)
        net = relu(net)
        return net


def conv_1x1(input, output_dim, weight_decay, name, bias=True):
    with tf.name_scope(name):
        return conv2d(input, output_dim, 1,1,1,1, weight_decay,stddev=0.02, name=name)

def pwise_block(input, output_dim, weight_decay, is_train, name, bias=True):
    with tf.name_scope(name):
        out=conv_1x1(input, output_dim, weight_decay, name, bias)
        out=batch_norm(out, train=is_train, name='pwb')
        out=relu(out)
        return out


def dwise_conv(input, k_h=3, k_w=3, channel_multiplier= 1, strides=[1,1,1,1],
               padding='SAME', weight_decay=0.0, stddev=0.02, name='dwise_conv'):
    with tf.variable_scope(name):
        in_channel=input.get_shape().as_list()[-1]
        w = tf.get_variable('w', [k_h, k_w, in_channel, channel_multiplier],
                        initializer=tf.truncated_normal_initializer(stddev=stddev),
                            regularizer=tf.contrib.layers.l2_regularizer(weight_decay))
        return tf.nn.depthwise_conv2d(input, w, strides, padding, rate=None,name=None,data_format=None)


def res_block(input, expansion_ratio, output_dim, stride, weight_decay, is_train, name, shortcut=True):
    with tf.name_scope(name), tf.variable_scope(name):
        # pw
        bottleneck_dim=round(expansion_ratio*input.get_shape().as_list()[-1])
        net = conv_1x1(input, bottleneck_dim, weight_decay, name='pw')
        net = batch_norm(net, train=is_train, name='pw')
        net = relu(net)
        # dw
        net = dwise_conv(net, strides=[1, stride, stride, 1], weight_decay=weight_decay, name='dw')
        net = batch_norm(net, train=is_train, name='dw')
        net = relu(net)
        # pw & linear
        net = conv_1x1(net, output_dim, weight_decay, name='pw_linear')
        net = batch_norm(net, train=is_train, name='pw_linear')

        # element wise add, only for stride==1
        if shortcut and stride == 1:
            in_dim=int(input.get_shape().as_list()[-1])
            if in_dim != output_dim:
                ins=conv_1x1(input, output_dim, weight_decay, name='ex_dim')
                net=ins+net
            else:
                net=input+net

        return net


def global_avg(x):
    with tf.name_scope('global_avg'):
        net=tf.layers.average_pooling2d(x, x.get_shape()[1:-1], 1)
        return net


def flatten(x):
    #flattened=tf.reshape(input,[x.get_shape().as_list()[0], -1])  # or, tf.layers.flatten(x)
    return tf.contrib.layers.flatten(x)