from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import math
import os
import random
import zipfile

import numpy as np
from six.moves import urllib
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

sess = tf.Session()
new_saver = tf.train.import_meta_graph('my-model.meta')
new_saver.restore(sess, tf.train.latest_checkpoint('./'))

embedding = tf.placeholder(tf.float32, [vocab_size, embedding_size])
batch_array = tf.placeholder(tf.float32, [batch_size, embedding_size])

normed_embedding = tf.nn.l2_normalize(embedding, dim=1)
normed_array = tf.nn.l2_normalize(batch_array, dim=1)

cosine_similarity = tf.matmul(normed_array, tf.transpose(normed_embedding, [1, 0]))

closest_words = tf.argmax(cosine_similarity, 1)  # shape [batch_size], type int64