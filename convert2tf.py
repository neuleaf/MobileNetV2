import os
from PIL import Image
import tensorflow as tf
from utils import *


FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('dataset', 'imagenet','The name of the dataset to convert.')
tf.app.flags.DEFINE_string('img_dir', './data/imagenet','The directory store images.')
tf.app.flags.DEFINE_string('train_datas', 'train.txt','The images and their labels')
tf.app.flags.DEFINE_string('output_dir', './tfrecords','Output directory where to store TFRecords files.')


def convert2example(img_dir, line):
    '''
    read
    :param path:
    :return:
    '''
    img_name, label=line.strip().split()
    img_path=os.path.join(img_dir, img_name)
    # for simplify, can use tf.gfile.FastGFile
    # img_data=tf.gfile.FastGFile(img_path, 'rb').read()
    # we need height and width

    image=Image.open(img_path)
    width, height = image.size

    # in case for gray image, convert to 3 channels RGB mode
    if image.mode != 'RGB':
        image=image.convert('RGB')

    img = image.resize((width, height))
    image_data=img.tobytes()

    example = tf.train.Example(features=tf.train.Features(feature={
        'image/encoded': bytes_feature(image_data),
        'image/height': int64_feature(height),
        'image/width': int64_feature(width),
        'image/label': int64_feature(int(label))
    }))

    return example



def main(_):
    if not tf.gfile.Exists(FLAGS.output_dir):
        tf.gfile.MakeDirs(FLAGS.output_dir)

    out_file=os.path.join(FLAGS.output_dir, FLAGS.dataset+'.tfrecord')

    num_samples=0
    with tf.python_io.TFRecordWriter(out_file) as tfWriter:
        with open(FLAGS.train_datas, 'r') as f:
            for line in f:
                print(line)
                example=convert2example(FLAGS.img_dir, line)
                tfWriter.write(example.SerializeToString())
                num_samples+=1
    print("Number of samples: {}".format(num_samples))


if __name__ == '__main__':
    tf.app.run()