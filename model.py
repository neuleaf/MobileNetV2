from ops import *
from utils import *
import os
import time


class MobileNetV2(object):
    def __init__(self, sess, dataset, epoch, batch_size, image_height, image_width, n_classes,
                 learning_rate, lr_decay, beta1, chkpt_dir, logs_dir, model_name, rand_crop=False, is_train=True):
        self.dataset=dataset
        self.model_name=model_name
        self.h=image_height
        self.w=image_width
        self.shape=[self.h, self.w]
        self.n_classes=n_classes
        self.epoch=epoch
        self.batch_size=batch_size
        self.learning_rate=learning_rate
        self.lr_decay=lr_decay
        self.train=is_train
        self.beta1=beta1
        self.sess=sess
        self.checkpoint_dir=chkpt_dir
        self.logs_dir=logs_dir
        self.rand_crop=rand_crop
        self.renew=True

    def _build_train_graph(self):
        self.x_=tf.placeholder(tf.float32, [None, self.h, self.w, 3], name='input')
        self.y_=tf.placeholder(tf.int64, [None], name='label')

        self.global_step = tf.Variable(0, name='global_step', trainable=False)

        logits, pred=self._nets(self.x_)

        # loss
        loss=tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.y_, logits=logits))

        vars=tf.get_collection(tf.GraphKeys.TABLE_INITIALIZERS, scope='mobilenetv2')
        update_ops=tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='mobilenetv2')

        # evaluate model, for classification
        correct_pred=tf.equal(tf.argmax(pred, 1), self.y_)
        acc=tf.reduce_mean(tf.cast(correct_pred, tf.float32))

        # learning rate decay
        lr_decay_step=self.dataset.shape[0] // self.batch_size # every epoch
        lr=tf.train.exponential_decay(self.learning_rate, global_step=self.global_step, decay_steps=lr_decay_step, decay_rate=self.lr_decay)
        # optimizer
        # tf.train.RMSPropOptimizer(learning_rate=self.lr, decay=0.9, momentum=0.9)
        self.train_op=tf.train.AdamOptimizer(learning_rate=lr, beta1=self.beta1).minimize(loss, global_step=self.global_step)

        # summary
        tf.summary.scalar('loss', loss)
        tf.summary.scalar('accuracy', acc)
        tf.summary.scalar('learning_rate', lr)
        self.summary_op = tf.summary.merge_all()

        # accesible points
        self.loss=loss
        self.acc=acc
        self.lr=lr

    def _nets(self, X, reuse=False):
        with tf.variable_scope('mobilenetv2', reuse=reuse):
            net = conv2d_block(input=X, out_dim=32, k=3, s=2, name='conv1_1', is_train=self.train)  # size/2

            net = res_block(net, expansion_ratio=6, out_put_dim=16, stride=1, name='res1_1', is_train=self.train, hyperlink=False)

            net = res_block(net, expansion_ratio=6, out_put_dim=24, stride=2, name='res2_1', is_train=self.train)  # size/4
            net = res_block(net, expansion_ratio=6, out_put_dim=24, stride=1, name='res2_2', is_train=self.train)

            net = res_block(net, expansion_ratio=6, out_put_dim=32, stride=2, name='res3_1', is_train=self.train)  # size/8
            net = res_block(net, expansion_ratio=6, out_put_dim=32, stride=1, name='res3_2', is_train=self.train)
            net = res_block(net, expansion_ratio=6, out_put_dim=32, stride=1, name='res3_3', is_train=self.train)

            net = res_block(net, expansion_ratio=6, out_put_dim=64, stride=1, name='res4_1', is_train=self.train)
            net = res_block(net, expansion_ratio=6, out_put_dim=64, stride=1, name='res4_2', is_train=self.train)
            net = res_block(net, expansion_ratio=6, out_put_dim=64, stride=1, name='res4_3', is_train=self.train)
            net = res_block(net, expansion_ratio=6, out_put_dim=64, stride=1, name='res4_4', is_train=self.train)

            net = res_block(net, expansion_ratio=6, out_put_dim=96, stride=2, name='res5_1', is_train=self.train)  # size/16
            net = res_block(net, expansion_ratio=6, out_put_dim=96, stride=1, name='res5_2', is_train=self.train)
            net = res_block(net, expansion_ratio=6, out_put_dim=96, stride=1, name='res5_3', is_train=self.train)

            net = res_block(net, expansion_ratio=6, out_put_dim=160, stride=2, name='res6_1', is_train=self.train)  # size/32
            net = res_block(net, expansion_ratio=6, out_put_dim=160, stride=1, name='res6_2', is_train=self.train)
            net = res_block(net, expansion_ratio=6, out_put_dim=160, stride=1, name='res6_3', is_train=self.train)

            net = res_block(net, expansion_ratio=6, out_put_dim=320, stride=1, name='res7_1', is_train=self.train)

            net = conv_1x1(net, output_dim=1280, name='conv8_1')
            net = global_avg(net)
            logits = conv_1x1(net, output_dim=self.n_classes, name='logits')

            pred=tf.nn.softmax(flatten(logits), name='prob')
            return flatten(logits), pred

    def load(self, checkpoint_dir):
        import re
        print(" [*] Reading checkpoints...")
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_name)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            counter = int(next(re.finditer("(\d+)(?!.*\d)", ckpt_name)).group(0))
            print(" [*] Success to read {}".format(ckpt_name))
            return True, counter
        else:
            print(" [*] Failed to find a checkpoint")
            return False, 0

    def _train(self):
        """train
        """
        sess=self.sess
        # initialize all variables
        init=tf.global_variables_initializer()
        sess.run(init)

        # saver for save/restore model
        saver=tf.train.Saver()

        # summary writer
        writer=tf.summary.FileWriter(self.logs_dir, self.sess.graph)

        # restore check-point if exists
        if not self.renew:
            could_load, _=self.load(self.checkpoint_dir)
            if could_load:
                print('load model from ', self.checkpoint_dir)
            else:
                print('No trained model to load.')

        start_epoch=0

        # how many batches DATA can be split into
        batch_idxs = self.dataset.shape[0] // self.batch_size
        # loop for epoch
        start_time=time.time()
        print('START TRAINING...')
        for epoch in range(start_epoch, self.epoch):
            start_batch_idx = 0
            # shuffle datas
            np.random.shuffle(self.dataset)

            for idx in range(start_batch_idx, batch_idxs):
                batch_files=self.dataset[idx*self.batch_size:(idx+1)*self.batch_size, :]
                x_files=batch_files[:,0]
                batch=[get_image(path, self.shape, rand_crop=self.rand_crop) for path in x_files]
                batch_images=np.array(batch).astype(np.float32)

                # here we don't need one hot label, because loss defined as tf.SPARSE_xxx_cross_entropy
                batch_labels=batch_files[:,1]

                feed_dict={self.x_:batch_images, self.y_:batch_labels}
                # train
                _, lr, step=sess.run([self.train_op, self.lr, self.global_step], feed_dict=feed_dict)

                # print logs and write summary
                if step % 20 == 0:
                    summ, loss, acc = sess.run([self.summary_op, self.loss, self.acc],
                                                       feed_dict=feed_dict)
                    writer.add_summary(summ, step)
                    print('epoch:{0}, global_step:{1}, batch_idx:{2}, time:{3:.3f}, lr:{4:.8f}, acc:{5:.6f}, loss:{6:.6f}'.format
                        (epoch, step, idx, time.time()-start_time, lr, acc, loss))

                # save model
                if np.mod(step, 500)==0:
                    saver.save(sess, os.path.join(self.checkpoint_dir, self.model_name), global_step=step)

        # save the last model when finish training
        save_path=saver.save(self.sess, os.path.join(self.checkpoint_dir, self.model_name))
        print('Final model saved in '+save_path)
        print('FINISHED TRAINING.')