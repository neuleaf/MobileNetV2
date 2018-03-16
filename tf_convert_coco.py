import tensorflow as tf
import json
import os
import numpy as np
from PIL import Image
import random
from dataset_tools.mscoco_label_map import COCO_MAP
from dataset_tools import dataset_util

flags = tf.app.flags
flags.DEFINE_string('data_dir', '', 'Root directory to raw COCO dataset.')
flags.DEFINE_string('img_dir', '', 'Images dir')
flags.DEFINE_string('ann_dir', '', 'Annotations dir')
flags.DEFINE_string('output_path', '', 'Path to output TFRecord')
flags.DEFINE_string('tfrecord_name', 'coco_train', 'generated tfrecord file prefix_name')
flags.DEFINE_boolean('ignore_difficult_instances', False,
                     'Whether to ignore difficult instances')

FLAGS=flags.FLAGS

RANDOM_SEED = 4242
SAMPLES_PER_FILES = 1000

def _process_image(data_dir, img_dir, ann_dir, img_name):
    '''
    get coco image, shape, bbox and label
    '''
    # read image
    img_file = os.path.join(data_dir, img_dir, img_name + '.jpg')
    image_data = tf.gfile.FastGFile(img_file, 'rb').read()
    # get shape
    img_ = Image.open(img_file)
    img = np.asarray(img_)
    if len(img.shape) == 2:
        c = 1
        h, w = img.shape
    else:
        h, w, c = img.shape
    shape = [h, w, c]

    # read annotation
    ann_file = os.path.join(data_dir, ann_dir, img_name + '.json')

    bboxes = []
    labels = []
    with open(ann_file, "r+") as f:
        allData = json.load(f)
        data = allData['annotation']
        print("read ready: ", img_name)
        for ann in data:
            label = COCO_MAP.label_idx[int(ann['category_id'])]
            labels.append(label)
            bbox = ann['bbox']
            xmin = float(bbox[0])
            ymin = float(bbox[1])
            xmax = xmin + float(bbox[2])
            ymax = ymin + float(bbox[3])
            bboxes.append((ymin / shape[0],
                           xmin / shape[1],
                           ymax / shape[0],
                           xmax / shape[1]
                           ))

    return image_data, shape, bboxes, labels


def _convert_to_example(image_data, labels, bboxes, shape):
    """Build an Example proto for an image example.

    Args:
      image_data: string, JPEG encoding of RGB image;
      labels: list of integers, identifier for the ground truth;
      labels_text: list of strings, human-readable labels;
      bboxes: list of bounding boxes; each box is a list of integers;
          specifying [xmin, ymin, xmax, ymax]. All boxes are assumed to belong
          to the same label as the image label.
      shape: 3 integers, image shapes in pixels.
    Returns:
      Example proto
    """
    xmin = []
    ymin = []
    xmax = []
    ymax = []
    for b in bboxes:
        assert len(b) == 4
        # pylint: disable=expression-not-assigned
        [l.append(point) for l, point in zip([ymin, xmin, ymax, xmax], b)]
        # pylint: enable=expression-not-assigned

    image_format = b'JPG'
    example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(shape[0]),
        'image/width': dataset_util.int64_feature(shape[1]),
        'image/channels': dataset_util.int64_feature(shape[2]),
        'image/shape': dataset_util.int64_feature(shape),
        'image/object/bbox/xmin': dataset_util.float_feature(xmin),
        'image/object/bbox/xmax': dataset_util.float_feature(xmax),
        'image/object/bbox/ymin': dataset_util.float_feature(ymin),
        'image/object/bbox/ymax': dataset_util.float_feature(ymax),
        'image/object/bbox/label': dataset_util.int64_feature(labels),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/encoded': dataset_util.bytes_feature(image_data)}))
    return example


def _add_to_tfrecord(data_dir, img_dir, ann_dir, img_name, tfrecord_writer):
    """Loads data from image and annotations files and add them to a TFRecord.

    Args:
      dataset_dir: Dataset directory;
      name: Image name to add to the TFRecord;
      tfrecord_writer: The TFRecord writer to use for writing.
    """
    image_data, shape, bboxes, labels = _process_image(data_dir, img_dir, ann_dir, img_name)
    example = _convert_to_example(image_data, labels, bboxes, shape)
    tfrecord_writer.write(example.SerializeToString())


def _get_output_filename(output_dir, name, idx):
    return '%s/%s_%05d.tfrecord' % (output_dir, name, idx)


def main(shuffling=False):
    """Runs the conversion operation.

    Args:
      dataset_dir: The dataset directory where the dataset is stored.
      output_dir: Output directory.
    """
    COCO_DIR=FLAGS.data_dir
    IMG_DIR =FLAGS.img_dir
    ANN_DIR =FLAGS.ann_dir

    if not tf.gfile.Exists(FLAGS.output_path):
        tf.gfile.MakeDirs(FLAGS.output_path)

    path = os.path.join(COCO_DIR, IMG_DIR)
    filenames = sorted(os.listdir(path))
    if shuffling:
        random.seed(RANDOM_SEED)
        random.shuffle(filenames)

    # Process dataset files.
    i = 0
    fidx = 0
    while i < len(filenames):
        # Open new TFRecord file.
        tf_filename = _get_output_filename(FLAGS.output_path, FLAGS.tfrecord_name, fidx)
        with tf.python_io.TFRecordWriter(tf_filename) as tfrecord_writer:
            j = 0
            while i < len(filenames) and j < SAMPLES_PER_FILES:
                print('Converting image %d/%d' % (i + 1, len(filenames)))
                filename = filenames[i]
                img_name = filename.split('.')[0]
                _add_to_tfrecord(COCO_DIR, IMG_DIR, ANN_DIR, tfrecord_writer)
                i += 1
                j += 1
            fidx += 1

    print('\nFinished converting the MSCOCO dataset!')


if __name__ == '__main__':
    tf.app.run()