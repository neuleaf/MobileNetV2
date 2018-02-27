from model import MobileNetV2
from utils import preprocess
import argparse
import tensorflow as tf
import os
import glob
import time
import numpy as np
from scipy.misc import imread, imresize


def parse_args():
    parser=argparse.ArgumentParser()
    parser.add_mutually_exclusive_group(required=False)

    parser.add_argument('--dataset_dir', type=str, default='./tfrecords', help='tfrecord file dir')
    parser.add_argument('--num_samples', type=int, help='the number of train samples')
    parser.add_argument('--epoch', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--n_classes', type=int)
    parser.add_argument('--image_height', type=int, default=224)
    parser.add_argument('--image_width', type=int, default=224)
    parser.add_argument('--is_train', dest='is_train', action='store_true')
    parser.add_argument('--no_train', dest='is_train', action='store_false')
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--lr_decay', type=float, default=0.98)
    parser.add_argument('--beta1', type=float, default=0.9)
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints')
    parser.add_argument('--logs_dir', type=str, default='logs')
    parser.add_argument('--model_name', type=str, default='mobileNetV2')
    parser.add_argument('--rand_crop', dest='rand_crop', action='store_true')
    parser.add_argument('--no_rand_crop', dest='rand_crop', action='store_false')
    parser.add_argument('--cpu', dest='cpu', action='store_true')
    parser.add_argument('--gpu', dest='cpu', action='store_false')
    # set default
    parser.set_defaults(is_train=True)
    parser.set_defaults(cpu=False)
    parser.set_defaults(rand_crop=False)

    return parser.parse_args()


def main():
    # parse arguments
    args=parse_args()

    if args.cpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    sess = tf.Session()

    if args.is_train:
        # read tfrecord files
        glob_pattern=os.path.join(args.dataset_dir, '*.tfrecord')
        tfrecords_list = glob.glob(glob_pattern)

        # check dirs
        if not os.path.exists(args.checkpoint_dir):
            os.makedirs(args.checkpoint_dir)
        if not os.path.exists(args.logs_dir):
            os.makedirs(args.logs_dir)

        model=MobileNetV2(sess=sess, tf_files=tfrecords_list, num_sampes=args.num_samples,
                      epoch=args.epoch, batch_size=args.batch_size,
                      image_height=args.image_height, image_width=args.image_width,
                      n_classes=args.n_classes,
                      is_train=args.is_train, learning_rate=args.learning_rate,
                      lr_decay=args.lr_decay,beta1=args.beta1,
                      chkpt_dir=args.checkpoint_dir, logs_dir=args.logs_dir,
                      model_name=args.model_name, rand_crop=args.rand_crop)
        model._build_train_graph()
        model._train()
    else:
        # restore model
        # saver = tf.train.import_meta_graph(
        #    os.path.join(args.checkpoint_dir,args.model_name+'-4000.meta'))
        # saver.restore(sess, tf.train.latest_checkpoint(args.checkpoint_dir))

        #
        model=MobileNetV2(sess=sess, tf_files='', num_sampes=args.num_samples,
                      epoch=args.epoch, batch_size=args.batch_size,
                      image_height=args.image_height, image_width=args.image_width,
                      n_classes=args.n_classes,
                      is_train=args.is_train, learning_rate=args.learning_rate,
                      lr_decay=args.lr_decay,beta1=args.beta1,
                      chkpt_dir=args.checkpoint_dir, logs_dir=args.logs_dir,
                      model_name=args.model_name, rand_crop=args.rand_crop)
        model._build_test_graph()
        saver=tf.train.Saver()
        ckpt = tf.train.get_checkpoint_state(args.checkpoint_dir)
        import re
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            saver.restore(sess, os.path.join(args.checkpoint_dir, ckpt_name))
            counter = int(next(re.finditer("(\d+)(?!.*\d)", ckpt_name)).group(0))
            print("[*] Success to read {}".format(ckpt_name))
        else:
            print("[*] Failed to find a checkpoint")

        # get input and output tensors from graph
        graph = tf.get_default_graph()
        input_x = graph.get_tensor_by_name("input:0")
        input_y = graph.get_tensor_by_name("label:0")
        prob = graph.get_tensor_by_name("mobilenetv2/prob:0")

        # prepare eval/test data and label
        img=imread('data/test/t_1_0.jpeg')
        img = imresize(img, (args.image_height, args.image_width))
        img=preprocess(img)
        print(img.dtype)
        label=1
        feed_dict={input_x:[img],input_y:[label]} # use [], because we need 4-D tensor

        start=time.time()
        res=sess.run(prob, feed_dict=feed_dict)[0] # index 0 for batch_size
        print('prob: {}, class: {}'.format(res, np.argmax(res)))
        print('time: {}'.format(time.time()-start))


if __name__=='__main__':
    main()