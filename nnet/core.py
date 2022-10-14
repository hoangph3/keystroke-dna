from tensorflow.keras import layers
import tensorflow as tf
from .layer import GlobalMeanStddevPooling1D, inception_module, residual_module, ArcMarginPenaltyLogists
from .Inception_ResNet_1DCNN import Inception_ResNet


def inception_resnet_masking(input_shape, embed_dims, embed_norm, num_classes=None, margin=0.5, scale=64, loss_fn=None, **kwargs):
    inputs = tf.keras.Input(shape=input_shape, name='inputs')
    labels = tf.keras.Input([], name="labels")
    x = layers.Masking(mask_value=0.)(inputs)

    x = inception_module(x, 64, 96, 128, 16, 32, 32)
    x = residual_module(x, 128)
    x = inception_module(x, 64, 96, 128, 16, 32, 32)
    x = residual_module(x, 128)
    x = inception_module(x, 64, 96, 128, 16, 32, 32)
    x = residual_module(x, 128)

    x = GlobalMeanStddevPooling1D()(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(embed_dims, name='latent', activation=None)(x)
    if embed_norm:
        outputs = layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1))(outputs)

    if loss_fn in ['triplet', 'contrastive']:
        model = tf.keras.Model(inputs=inputs, outputs=outputs, name="inception_resnet_masking")
    elif loss_fn == 'arcface':
        outputs = ArcMarginPenaltyLogists(num_classes=num_classes, margin=margin, logist_scale=scale)([outputs, labels])
        model = tf.keras.Model(inputs=(inputs, labels), outputs=outputs, name="inception_resnet_masking")
    else:
        raise NotImplementedError

    return model


def inception_resnet(input_shape, embed_dims, embed_norm, **kwargs):
    inputs = tf.keras.Input(shape=input_shape, name='inputs')
    x = inputs

    x = inception_module(x, 64, 96, 128, 16, 32, 32)
    x = residual_module(x, 128)
    x = inception_module(x, 64, 96, 128, 16, 32, 32)
    x = residual_module(x, 128)
    x = inception_module(x, 64, 96, 128, 16, 32, 32)
    x = residual_module(x, 128)

    x = GlobalMeanStddevPooling1D()(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(embed_dims, name='latent', activation=None)(x)
    if embed_norm:
        outputs = layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1))(outputs)
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="inception_resnet")
    return model


def lstm(input_shape, embed_dims, embed_norm, **kwargs):
    inputs = tf.keras.Input(shape=input_shape, name='inputs')
    x = layers.Masking(mask_value=0.)(inputs)

    x = layers.BatchNormalization()(x)
    x = layers.LSTM(128, return_sequences=True)(x)
    x = layers.Dropout(0.2)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LSTM(128)(x)

    outputs = layers.Dense(embed_dims, name='latent', activation=None)(x)
    if embed_norm:
        outputs = layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1))(outputs)
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="lstm")
    return model
