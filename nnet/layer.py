import tensorflow as tf
from tensorflow.keras import layers
import math


class GlobalMeanStddevPooling1D(tf.keras.layers.Layer):
    """
    Compute arithmetic mean and standard deviation of the inputs along the time steps dimension,
    then output the concatenation of the computed stats.
    Inputs shape = (Batch, Time, Channels)
    """
    def call(self, inputs):
        TIME_AXIS = 1
        means = tf.math.reduce_mean(inputs, axis=TIME_AXIS, keepdims=True)
        variances = tf.math.reduce_mean(tf.math.square(inputs - means), axis=TIME_AXIS)
        means = tf.squeeze(means, TIME_AXIS)
        stddevs = tf.math.sqrt(tf.clip_by_value(variances, 1e-10, variances.dtype.max))
        return tf.concat((means, stddevs), axis=TIME_AXIS)


def inception_module(layer_in, f1, f2_in, f2_out, f3_in, f3_out, f4_out):
    # 1x1 conv
    conv1 = layers.Conv1D(f1, 1, padding='same', activation='leaky_relu')(layer_in)

    # 3x3 conv
    conv3 = layers.Conv1D(f2_in, 1, padding='same')(layer_in)
    conv3 = layers.BatchNormalization()(conv3)
    conv3 = layers.Activation(activation='leaky_relu')(conv3)
    conv3 = layers.Conv1D(f2_out, 3, padding='same')(conv3)
    conv3 = layers.BatchNormalization()(conv3)
    conv3 = layers.Activation(activation='leaky_relu')(conv3)

    # 5x5 conv
    conv5 = layers.Conv1D(f3_in, 1, padding='same')(layer_in)
    conv5 = layers.BatchNormalization()(conv5)
    conv5 = layers.Activation(activation='leaky_relu')(conv5)
    conv5 = layers.Conv1D(f3_out, 5, padding='same')(conv5)
    conv5 = layers.BatchNormalization()(conv5)
    conv5 = layers.Activation(activation='leaky_relu')(conv5)

    # 3x3 max pooling
    pool = layers.MaxPooling1D(3, strides=1, padding='same')(layer_in)
    pool = layers.Conv1D(f4_out, 1, padding='same')(pool)
    pool = layers.BatchNormalization()(pool)
    pool = layers.Activation(activation='leaky_relu')(pool)

    # concatenate filters, assumes filters/channels last
    layer_out = layers.Concatenate(axis=-1)([conv1, conv3, conv5, pool])
    return layer_out


def residual_module(layer_in, n_filters):
    merge_input = layer_in

    # check if the number of filters needs to be increase, assumes channels last format
    if layer_in.shape[-1] != n_filters:
        merge_input = layers.Conv1D(n_filters, 1, padding='same', activation='leaky_relu',
                                    kernel_initializer='he_normal')(layer_in)

    # conv1
    conv1 = layers.Conv1D(n_filters, 3, padding='same', activation='leaky_relu', kernel_initializer='he_normal')(
        layer_in)

    # conv2
    conv2 = layers.Conv1D(n_filters, 3, padding='same', activation='linear', kernel_initializer='he_normal')(conv1)

    # add filters, assumes filters/channels last
    layer_out = layers.Add()([conv2, merge_input])

    # activation function
    layer_out = layers.LeakyReLU()(layer_out)

    return layer_out


class ArcMarginPenaltyLogists(tf.keras.layers.Layer):
    """ArcMarginPenaltyLogists"""
    def __init__(self, num_classes, margin=0.5, logist_scale=64, **kwargs):
        super(ArcMarginPenaltyLogists, self).__init__(**kwargs)
        self.num_classes = num_classes
        self.margin = margin
        self.logist_scale = logist_scale

    def build(self, input_shape):
        self.w = self.add_variable(
            "weights", shape=[input_shape[0][-1], self.num_classes])
        self.cos_m = tf.identity(math.cos(self.margin), name='cos_m')
        self.sin_m = tf.identity(math.sin(self.margin), name='sin_m')
        self.th = tf.identity(math.cos(math.pi - self.margin), name='th')
        self.mm = tf.multiply(self.sin_m, self.margin, name='mm')

    def call(self, inputs):
        embds, labels = inputs
        normed_embds = tf.nn.l2_normalize(embds, axis=1, name='normed_embd')
        normed_w = tf.nn.l2_normalize(self.w, axis=0, name='normed_weights')

        cos_t = tf.matmul(normed_embds, normed_w, name='cos_t')
        sin_t = tf.sqrt(1. - cos_t ** 2, name='sin_t')

        cos_mt = tf.subtract(
            cos_t * self.cos_m, sin_t * self.sin_m, name='cos_mt')

        cos_mt = tf.where(cos_t > self.th, cos_mt, cos_t - self.mm)

        mask = tf.one_hot(tf.cast(labels, tf.int32), depth=self.num_classes,
                          name='one_hot_mask')

        logists = tf.where(mask == 1., cos_mt, cos_t)
        logists = tf.multiply(logists, self.logist_scale, 'arcface_logist')

        return logists

    def get_config(self):
        config = super().get_config()
        config.update({"num_classes": self.num_classes,
                       "margin": self.margin,
                       "logist_scale": self.logist_scale})
        return config
