# Copyright 2017 Zheng Zhuo. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Convert the UC Merced landuse dataset to TFRecords of TF-Example protos.
This module reads
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gzip
import os
import sys
from skimage.io import imread
from skimage.transform import resize
import numpy as np
import shutil
from six.moves import urllib
import tensorflow as tf

from datasets import dataset_utils

DATASET = r"E:\SceneClassificationBasedOnDeepLearningForHighSpatialResolutionRemoteSensingImages\Data\UCM\Images"
TF_RECORD_SET = r"E:\tmps"

SAMPLES_PER_CLASS = 100
SAMPLES_PER_FILES = 10
TRAIN_RATIO = 0.8

_IMAGE_SIZE = 256
_NUM_CHANNELS = 3

_CLASS_NAMES = [
    'agricultural',
    'airplane',
    'baseballdiamond',
    'beach',
    'buildings',
    'chaparral',
    'denseresidential',
    'forest',
    'freeway',
    'golfcourse',
    'harbor',
    'intersection',
    'mediumresidential',
    'mobilehomepark',
    'overpass',
    'parkinglot',
    'river',
    'runway',
    'sparseresidential',
    'storagetanks',
    'tenniscourt'
]


def _extract_images(filenames):
    """Extract the images into a numpy array.

    Args:
      filenames: The path to an UC Merced images file.

    Returns:
      datas:A numpy array of shape [number_of_images, height, width, channels].
      n: number of images
    """
    print('Extracting images from: ', filenames)
    datas = []
    for fn in filenames:
        img = imread(fn)
        img = resize(img, (_IMAGE_SIZE, _IMAGE_SIZE, _NUM_CHANNELS))
        datas.append(img)
    datas = np.asarray(datas)
    n, h, w, c = datas.shape
    assert h == _IMAGE_SIZE and w == _IMAGE_SIZE and c == _NUM_CHANNELS, "Those shapes of images are not standard."
    return datas, n


def _add_to_tfrecord(img_filenames, type, tfrecord_writer):
    """Loads data from the binary MNIST files and writes files to a TFRecord.

    Args:
      img_filenames: The filename of the UC Merced images.
      type: The category of those images.
      tfrecord_writer: The TFRecord writer to use for writing.
    """
    images, n = _extract_images(img_filenames)
    labels = np.zeros((n,), np.int64)
    labels.fill(type)
    shape = (_IMAGE_SIZE, _IMAGE_SIZE, _NUM_CHANNELS)
    with tf.Graph().as_default():
        image = tf.placeholder(dtype=tf.uint8, shape=shape)
        encoded_png = tf.image.encode_png(image)

        with tf.Session('') as sess:
            for j in range(n):
                sys.stdout.write('\r>> Converting image %d/%d\n' % (j + 1, n))
                sys.stdout.flush()

                png_string = sess.run(encoded_png, feed_dict={image: images[j]})

                example = dataset_utils.image_to_tfexample(
                    png_string, 'png'.encode(), _IMAGE_SIZE, _IMAGE_SIZE, labels[j])
                tfrecord_writer.write(example.SerializeToString())


def _get_output_filename(tfrecord_dir, type, id):
    """Creates the output filename.

    Args:
      tfrecord_dir: The directory where the tf-record files are stored.
      type: The category of ucm images.
      id: a number for a specific tf-record file.

    Returns:
      An absolute file path.
    """
    return '{0}/{1}/ucm_{2}_{3}.tfrecord'.format(tfrecord_dir, type, type, id)


def split_into_train_test(train_ratio):
    num = int(100 / SAMPLES_PER_FILES * train_ratio)
    tf_dirs = os.listdir(TF_RECORD_SET)

    train_dir_name = "{0}/train".format(TF_RECORD_SET)
    test_dir_name = "{0}/test".format(TF_RECORD_SET)
    if not os.path.exists(train_dir_name):
        os.makedirs(train_dir_name)
    if not os.path.exists(test_dir_name):
        os.makedirs(test_dir_name)

    for tf_dir in tf_dirs:
        abs_tf_dir = os.path.join(TF_RECORD_SET, tf_dir)
        fns = os.listdir(abs_tf_dir)
        # move files to directory for training
        for filename in fns[:num]:
            shutil.move(os.path.join(abs_tf_dir, filename), os.path.join(train_dir_name, filename))
            print("{0}->train".format(filename))
        # move files to directory for testing
        for filename in fns[num:]:
            shutil.move(os.path.join(abs_tf_dir, filename), os.path.join(test_dir_name, filename))
            print("{0}->test".format(filename))
    # clear redundant directory
    for cn in _CLASS_NAMES:
        delete_dir = os.path.join(TF_RECORD_SET, cn)
        os.removedirs(delete_dir)


def run(dataset_dir, is_split_into_train_test=False):
    """Runs the conversion operation.

    Args:
      dataset_dir: The dataset directory where the dataset is stored.
    """
    for cn in _CLASS_NAMES:
        dir_pattern = "{0}/{1}"
        insance = dir_pattern.format(dataset_dir, cn)
        if not tf.gfile.Exists(insance):
            tf.gfile.MakeDirs(insance)

    for idx, type in enumerate(_CLASS_NAMES):
        dir = os.path.join(DATASET, type)
        img_filenames = os.listdir(dir)
        img_filenames = list(map(lambda e: os.path.join(DATASET, type, e), img_filenames))
        for i in range(int(len(img_filenames) / SAMPLES_PER_FILES)):
            partial = img_filenames[i * SAMPLES_PER_FILES:i * SAMPLES_PER_FILES + SAMPLES_PER_FILES]
            with tf.python_io.TFRecordWriter(_get_output_filename(TF_RECORD_SET, type, i)) as tfrecord_writer:
                _add_to_tfrecord(partial, idx, tfrecord_writer)

    if is_split_into_train_test:
        split_into_train_test(TRAIN_RATIO)
    # # Finally, write the labels file:
    # labels_to_class_names = dict(zip(range(len(_CLASS_NAMES)), _CLASS_NAMES))
    # dataset_utils.write_label_file(labels_to_class_names, dataset_dir)

    print('\nFinished converting the UC Merced dataset!')


if __name__ == "__main__":
    run(TF_RECORD_SET, True)
