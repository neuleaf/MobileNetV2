import numpy as np
import tensorflow as tf
from scipy.misc import imread, imresize, imsave


def preprocess(image):
    img = 2.0 / 255.0 * image - 1.0
    return img

# tfrecord example features
def int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def int64_list_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def bytes_list_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def float_list_feature(value):
  return tf.train.Feature(float_list=tf.train.FloatList(value=value))


# read tf_record
def read_tfrecord(filename_queue):
    feature = {'image/encoded': tf.FixedLenFeature([], tf.string),
               'image/height': tf.FixedLenFeature([], tf.int64),
               'image/width': tf.FixedLenFeature([], tf.int64),
               'image/label': tf.FixedLenFeature([], tf.int64)}

    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)

    features = tf.parse_single_example(serialized_example, features=feature)

    image  = tf.decode_raw(features['image/encoded'], tf.uint8)
    height = tf.cast(features['image/height'],tf.int32)
    width  = tf.cast(features['image/width'], tf.int32)
    label  = tf.cast(features['image/label'], tf.int32)
    img = tf.reshape(image, [height, width, 3])

    # preprocess
    # center_crop
    img = tf.image.resize_images(img, [256, 256])
    j = int(round((256 - 224) / 2.))
    i = int(round((256 - 224) / 2.))
    img = img[j:j+224, i:i+224, :]

    img = tf.cast(img, tf.float32) * (2. / 255) - 1.0

    return img, label

def get_batch(infile, batch_size, num_threads=4, shuffle=False, min_after_dequeue=None):
    # 使用batch，img的shape必须是静态常量
    image, label = read_tfrecord(infile)

    if min_after_dequeue is None:
        min_after_dequeue = batch_size * 10
    capacity = min_after_dequeue + 3 * batch_size

    if shuffle:
        img_batch, label_batch = tf.train.shuffle_batch([image, label], batch_size=batch_size,
                                                    capacity=capacity,num_threads=num_threads,
                                                    min_after_dequeue=min_after_dequeue)
    else:
        img_batch, label_batch = tf.train.batch([image, label], batch_size,
                                                capacity=capacity, num_threads=num_threads,
                                                allow_smaller_final_batch=True)

    return img_batch, label_batch