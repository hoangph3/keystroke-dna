import tensorflow as tf
import numpy as np
import os


def save_tfrecord(data, label, filepath):
    with tf.io.TFRecordWriter(filepath) as writer:
        for i in range(len(data)):
            features = tf.train.Features(
                feature={
                    "data": tf.train.Feature(bytes_list=tf.train.BytesList(value=[data[i].astype(np.float32).tobytes()])),
                    "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[label[i]]))
                }
            )
            example = tf.train.Example(features=features)
            serialized = example.SerializeToString()
            writer.write(serialized)
    return


def load_tfrecord(dirname):
    filenames = [os.path.join(dirname, filename) for filename in sorted(os.listdir(dirname))]
    dataset = tf.data.TFRecordDataset(filenames, num_parallel_reads=4)
    print("Load dataset contains {} records".format(len(filenames)))
    return dataset
