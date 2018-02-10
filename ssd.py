from ops import *
from utils import *
from ssd.utils import *
import os
import time


class SSD(object):
    def __init__(self):
        self.num_classes = 91
        self.feat_layers=['block6', 'block7', 'block8', 'block9', 'block10', 'block11']
        self.feat_shapes = [(38, 38), (19, 19), (10, 10), (5, 5), (3, 3), (1, 1)]
        self.anchor_size_bounds=[0.15, 0.90],
        # anchor_size_bounds=[0.20, 0.90],
        self.anchor_sizes=[(21., 45.),
                      (45., 99.),
                      (99., 153.),
                      (153., 207.),
                      (207., 261.),
                      (261., 315.)],
        # anchor_sizes=[(30., 60.),
        #               (60., 111.),
        #               (111., 162.),
        #               (162., 213.),
        #               (213., 264.),
        #               (264., 315.)],
        self.anchor_ratios=[[2, .5],
                       [2, .5, 3, 1./3],
                       [2, .5, 3, 1./3],
                       [2, .5, 3, 1./3],
                       [2, .5],
                       [2, .5]],
        self.anchor_steps=[8, 16, 32, 64, 100, 300],
        self.anchor_offset=0.5,
        self.normalizations=[20, -1, -1, -1, -1, -1],
        self.prior_scaling=[0.1, 0.1, 0.2, 0.2]

    def _build_graph(self):
        x=tf.placeholder(tf.float32, shape=[], name='inputs')
        gclasses=tf.placeholder()
        glocalisations=tf.placeholder()
        gscores=tf.placeholder()

        predictions, localisations, logits, end_points = self._nets(x)

        # loss
        conf_pos_loss, conf_neg_loss, loc_loss=ssd_losses(logits, localisations, gclasses, glocalisations, gscores)
        total_loss = conf_pos_loss + conf_neg_loss + loc_loss

        # optimizer
        optim= tf.train.AdamOptimizer(learning_rate=self.lr,beta1=self.beta1).minimize(total_loss)



    def _nets(self, X, reuse):
        w_d=self.weight_decay
        exp=6 # expansion ratio
        is_train=self.train
        end_points = {}
        with tf.variable_scope('mobilenetv2_ssd', reuse=reuse):
            net = conv2d_block(X, 32, 3, 2, w_d, is_train, name='conv1_1')
            end_points['block1'] = net

            net = res_block(net, exp, 16, 1, w_d, is_train, name='res2_1')
            end_points['block2'] = net

            net = res_block(net, exp, 24, 2, w_d, is_train, name='res3_1')  # size/4
            net = res_block(net, exp, 24, 1, w_d, is_train, name='res3_2')
            end_points['block3'] = net

            net = res_block(net, exp, 32, 2, w_d, is_train, name='res4_1')  # size/8
            net = res_block(net, exp, 32, 1, w_d, is_train, name='res4_2')
            net = res_block(net, exp, 32, 1, w_d, is_train, name='res4_3')
            end_points['block4'] = net

            net = res_block(net, exp, 64, 1, w_d, is_train, name='res5_1')
            net = res_block(net, exp, 64, 1, w_d, is_train, name='res5_2')
            net = res_block(net, exp, 64, 1, w_d, is_train, name='res5_3')
            net = res_block(net, exp, 64, 1, w_d, is_train, name='res5_4')
            end_points['block5'] = net

            net = res_block(net, exp, 96, 2, w_d, is_train, name='res6_1')  # size/16
            net = res_block(net, exp, 96, 1, w_d, is_train, name='res6_2')
            net = res_block(net, exp, 96, 1, w_d, is_train, name='res6_3')
            end_points['block6'] = net

            net = res_block(net, exp, 160, 2, w_d, is_train, name='res7_1')  # size/32
            net = res_block(net, exp, 160, 1, w_d, is_train, name='res7_2')
            net = res_block(net, exp, 160, 1, w_d, is_train, name='res7_3')
            end_points['block7'] = net

            end_point = 'block8'
            with tf.variable_scope(end_point):
                net = separable_conv(net, 3, 512, 1, w_d, name='sep8_1', pad='SAME')  # size/32
            end_points[end_point] = net

            end_point = 'block9'
            with tf.variable_scope(end_point):
                net = separable_conv(net, 3, 256, 2, w_d, name='sep9_1', pad='SAME')  # size/64
            end_points[end_point] = net

            end_point = 'block10'
            with tf.variable_scope(end_point):
                net = separable_conv(net, 3, 256, 2, w_d, name='sep10_1', pad='SAME')
            end_points[end_point] = net

            end_point = 'block11'
            with tf.variable_scope(end_point):
                net = separable_conv(net, 3, 256, 2, w_d, name='sep11_1', pad='VALID')
            end_points[end_point] = net

            # Prediction and localisations layers.
            predictions = []
            logits = []
            localisations = []
            for i, layer in enumerate(self.feat_layers):
                with tf.variable_scope(layer + '_box'):
                    p, l = ssd_multibox_layer(end_points[layer],
                                              self.num_classes,
                                              self.anchor_sizes[i],
                                              self.anchor_ratios[i],
                                              self.normalizations[i])
                predictions.append(slim.softmax(p))
                logits.append(p)
                localisations.append(l)

            return predictions, localisations, logits, end_points

    def _train(self):
