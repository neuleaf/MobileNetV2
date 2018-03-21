import tensorflow as tf
from ops import *
from ssd.utils import *
from ssd.config import *

def ssd_net(inputs):
    end_points = {}
    exp = 6  # expansion ratio
    w_d = 0.0
    is_train = True

    with tf.variable_scope('mobilenetv2'):
        # Block 1.
        net = conv2d_block(inputs, 32, 3, 2, is_train, name='conv1_1')  # size/2, 256
        end_points['block1'] = net

        # Block 2.
        net = res_block(net, 1, 16, 1, is_train, name='res2_1')
        end_points['block2'] = net

        # Block 3.
        net = res_block(net, exp, 24, 2, is_train, name='res3_1')  # size/4, 128
        net = res_block(net, exp, 24, 1, is_train, name='res3_2')
        end_points['block3'] = net

        # Block 4.
        net = res_block(net, exp, 32, 2, is_train, name='res4_1')  # size/8, 64
        net = res_block(net, exp, 32, 1, is_train, name='res4_2')
        net = res_block(net, exp, 32, 1, is_train, name='res4_3')
        end_points['block4'] = net

        # Block 5.
        net = res_block(net, exp, 64, 1, is_train, name='res5_1')
        net = res_block(net, exp, 64, 1, is_train, name='res5_2')
        net = res_block(net, exp, 64, 1, is_train, name='res5_3')
        net = res_block(net, exp, 64, 1, is_train, name='res5_4')
        end_points['block5'] = net

        # Block 6.
        net = res_block(net, exp, 96, 2, is_train, name='res6_1')  # size/16, 32
        net = res_block(net, exp, 96, 1, is_train, name='res6_2')
        net = res_block(net, exp, 96, 1, is_train, name='res6_3')
        end_points['block6'] = net

        # Block 7.
        net = res_block(net, exp, 160, 2, is_train, name='res7_1')  # size/32, 16
        net = res_block(net, exp, 160, 1, is_train, name='res7_2')
        net = res_block(net, exp, 160, 1, is_train, name='res7_3')
        end_points['block7'] = net

        end_point = 'block8'
        with tf.variable_scope(end_point):
            net = pad2d(net, pad=(1, 1))
            net = separable_conv(net, 3, 512, 2, name='sep8_1', pad='VALID')  # size/64, 8
        end_points[end_point] = net
        end_point = 'block9'
        with tf.variable_scope(end_point):
            net = pad2d(net, pad=(1, 1))
            net = separable_conv(net, 3, 256, 2, name='sep9_1', pad='VALID')  # size/128, 4
        end_points[end_point] = net
        end_point = 'block10'
        with tf.variable_scope(end_point):
            net = pad2d(net, pad=(1, 1))
            net = separable_conv(net, 3, 256, 2, name='sep10_1', pad='VALID')  # size/256, 2
        end_points[end_point] = net
        end_point = 'block11'
        with tf.variable_scope(end_point):
            net = pad2d(net, pad=(1, 1))
            net = separable_conv(net, 4, 256, 1, name='sep11_1', pad='VALID')  # size/512, 1
        end_points[end_point] = net

        # Prediction and localisations layers.
        predictions = []
        logits = []
        localisations = []
        for i, layer in enumerate(feat_layers):
            with tf.variable_scope(layer + '_box'):
                p, l = ssd_multibox_layer(end_points[layer],
                                                      num_classes,
                                                      anchor_sizes[i],
                                                      anchor_ratios[i],
                                                      normalizations[i])
            predictions.append(tf.nn.softmax(p))
            logits.append(p)
            localisations.append(l)

        return predictions, localisations, logits, end_points