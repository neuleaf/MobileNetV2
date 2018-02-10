from ops import *
from utils import *
import os
import time


class MobileNetV2(object):
    def __init__(self, sess, tf_files, num_sampes, epoch, batch_size, image_height, image_width, n_classes,
                 learning_rate, lr_decay, beta1, chkpt_dir, logs_dir, model_name, rand_crop=False, is_train=True):
        self.tf_files=tf_files # tfrecord list
        self.num_samples=num_sampes
        self.model_name=model_name
        self.h=image_height
        self.w=image_width
        self.shape=[self.h, self.w]
        self.n_classes=n_classes
        self.epoch=epoch
        self.batch_size=batch_size
        self.learning_rate=learning_rate
        self.lr_decay=lr_decay
        self.weight_decay=0.00004
        self.train=is_train
        self.beta1=beta1
        self.sess=sess
        self.checkpoint_dir=chkpt_dir
        self.logs_dir=logs_dir
        self.rand_crop=rand_crop
        self.renew=False

    def _build_train_graph(self):
        self.x_=tf.placeholder(tf.float32, [None, self.h, self.w, 3], name='input')
        self.y_=tf.placeholder(tf.int64, [None], name='label')

        self.global_step = tf.Variable(0, name='global_step', trainable=False)

        logits, pred=self._nets(self.x_)

        # loss
        loss_=tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.y_, logits=logits))
        # L2 regularization
        l2_loss=tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables()
                          if 'bn' not in v.name and 'bias' not in v.name])
        loss=loss_ + self.weight_decay*l2_loss

        # evaluate model, for classification
        correct_pred=tf.equal(tf.argmax(pred, 1), self.y_)
        acc=tf.reduce_mean(tf.cast(correct_pred, tf.float32))

        # learning rate decay
        lr_decay_step=self.num_samples // self.batch_size # every epoch
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

    def build_test_graph(self):
        self.x_ = tf.placeholder(tf.float32, [None, self.h, self.w, 3], name='input')
        self.y_ = tf.placeholder(tf.int64, [None], name='label')
        _, _ = self._nets(self.x_)

    def _nets(self, X, reuse=False):
        exp=6 # expansion ratio
        is_train=self.train
        with tf.variable_scope('mobilenetv2', reuse=reuse):
            net = conv2d_block(X, 32, 3, 2, is_train, name='conv1_1')  # size/2

            net = res_block(net, exp, 16, 1, is_train, name='res2_1')

            net = res_block(net, exp, 24, 2, is_train, name='res3_1')  # size/4
            net = res_block(net, exp, 24, 1, is_train, name='res3_2')

            net = res_block(net, exp, 32, 2, is_train, name='res4_1')  # size/8
            net = res_block(net, exp, 32, 1, is_train, name='res4_2')
            net = res_block(net, exp, 32, 1, is_train, name='res4_3')

            net = res_block(net, exp, 64, 1, is_train, name='res5_1')
            net = res_block(net, exp, 64, 1, is_train, name='res5_2')
            net = res_block(net, exp, 64, 1, is_train, name='res5_3')
            net = res_block(net, exp, 64, 1, is_train, name='res5_4')

            net = res_block(net, exp, 96, 2, is_train, name='res6_1')  # size/16
            net = res_block(net, exp, 96, 1, is_train, name='res6_2')
            net = res_block(net, exp, 96, 1, is_train, name='res6_3')

            net = res_block(net, exp, 160, 2, is_train, name='res7_1')  # size/32
            net = res_block(net, exp, 160, 1, is_train, name='res7_2')
            net = res_block(net, exp, 160, 1, is_train, name='res7_3')

            net = res_block(net, exp, 320, 1,  is_train, name='res8_1')

            net = pwise_block(net, 1280,  is_train, name='conv9_1')
            net = global_avg(net)
            logits = flatten(conv_1x1(net, self.n_classes,  name='logits'))

            pred=tf.nn.softmax(logits, name='prob')
            return logits, pred

    def load(self, saver, checkpoint_dir):
        import re
        print("[*] Reading checkpoints...")
        checkpoint_dir = os.path.join(checkpoint_dir)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            counter = int(next(re.finditer("(\d+)(?!.*\d)", ckpt_name)).group(0))
            print("[*] Success to read {}".format(ckpt_name))
            return True, counter
        else:
            print("[*] Failed to find a checkpoint")
            return False, 0

    def _train(self):
        """train
        """
        sess=self.sess

        # saver for save/restore model
        saver=tf.train.Saver()

        # summary writer
        writer=tf.summary.FileWriter(self.logs_dir, self.sess.graph)

        # restore check-point if exists
        if not self.renew:
            print('[*] Try to load trained model...')
            could_load, step=self.load(saver, self.checkpoint_dir)
            if could_load:
                tf.assign(self.global_step, step)

        total_step = int(self.num_samples / self.batch_size * self.epoch)

        # read queue
        filename_queue = tf.train.string_input_producer(self.tf_files, num_epochs=None)
        img_batch, label_batch = get_batch(filename_queue, self.batch_size)

        # init
        init = tf.global_variables_initializer()
        sess.run(init)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        print('START TRAINING...')
        start_time = time.time()
        while not coord.should_stop() and self.global_step.eval(session=sess) < total_step:
            batch_images, batch_labels=sess.run([img_batch, label_batch])
            feed_dict={self.x_:batch_images, self.y_:batch_labels}
             # train
            _, lr, step=sess.run([self.train_op, self.lr, self.global_step], feed_dict=feed_dict)

            # print logs and write summary
            if step % 10 == 0:
                summ, loss, acc = sess.run([self.summary_op, self.loss, self.acc],
                                                    feed_dict=feed_dict)
                writer.add_summary(summ, step)
                print('global_step:{0}, time:{1:.3f}, lr:{2:.8f}, acc:{3:.6f}, loss:{4:.6f}'.format
                    (step, time.time()-start_time, lr, acc, loss))

            # validation
            # TODO

            # save model
            if step % 500 == 0:
                    save_path=saver.save(sess, os.path.join(self.checkpoint_dir, self.model_name), global_step=step)
                    print('Current model saved in '+save_path)


        '''
        # how many batches DATA can be split into
        batch_idxs = self.dataset.shape[0] // self.batch_size
        # loop for epoch
        
        for epoch in range(0, self.epoch):
            start_batch_idx = 0
            # shuffle datas
            np.random.shuffle(self.dataset)

            for idx in range(start_batch_idx, batch_idxs):
                batch_files=self.dataset[idx*self.batch_size:(idx+1)*self.batch_size]
                x_files=batch_files[:,0]
                batch=[get_image(path, self.shape, rand_crop=self.rand_crop) for path in x_files]
                batch_images=np.array(batch).astype(np.float32)

                # here we don't need one hot label, because loss defined as tf.SPARSE_xxx_cross_entropy
                batch_labels=[int(l) for l in batch_files[:,1]]

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

                # validation
                # TODO

                # save model
                if np.mod(step, 500)==0:
                    saver.save(sess, os.path.join(self.checkpoint_dir, self.model_name), global_step=step)
        '''

        # save the last model when finish training
        save_path=saver.save(self.sess, os.path.join(self.checkpoint_dir, self.model_name), global_step= step)
        print('Final model saved in '+save_path)
        print('FINISHED TRAINING.')