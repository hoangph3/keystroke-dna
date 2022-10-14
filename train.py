import os
import random
import numpy as np

from tensorflow import keras
from tensorflow.keras import optimizers
from tensorflow.keras import Model
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
import tensorflow as tf

from util.hparams import get_hparams
from util.logger import get_logger
from nnet import core
from util.dataio import load_tfrecord
from nnet.loss import SupervisedContrastiveLoss
import tensorflow_addons as tfa
from util.config import config_gpus
from util.callback import EqualErrorRate


def main():
    config_gpus()

    # get hparams
    hps = get_hparams()
    logger = get_logger(hps.model_dir)
    logger.info(hps)

    # get model
    if os.path.exists(hps.train.checkpoint_dir):
        model: Model = keras.models.load_model(hps.train.checkpoint_dir)
        logger.info("Loaded checkpoint from {}".format(hps.train.checkpoint_dir))
    else:
        logger.info("Training model from scratch at {}".format(hps.model_dir))
        if hps.model.name == "inception_resnet_masking":
            model = core.inception_resnet_masking(**hps.model)
        elif hps.model.name == "inception_resnet":
            model = core.inception_resnet(**hps.model)
        elif hps.model.name == "inception_resnet_v2":
            model = core.Inception_ResNet(length=hps.model.input_shape[0],
                                          num_channel=hps.model.input_shape[1],
                                          num_filters=128,
                                          dropout_rate=0.4,
                                          output_nums=hps.model.embed_dims,
                                          normalize=hps.model.embed_norm).Inception_ResNet_v2()
        elif hps.model.name == "lstm":
            model = core.lstm(**hps.model)
        else:
            raise NotImplementedError
    model.summary(print_fn=logger.info)

    # compile model
    lr_schedule = optimizers.schedules.ExponentialDecay(initial_learning_rate=hps.train.learning_rate,
                                                        decay_steps=hps.train.decay_steps,
                                                        decay_rate=hps.train.decay_rate)
    optimizer = optimizers.Adagrad(learning_rate=lr_schedule)
    if hps.model.loss_fn == "contrastive":
        loss_fn = SupervisedContrastiveLoss(temperature=hps.model.margin)
    elif hps.model.loss_fn == "triplet":
        loss_fn = tfa.losses.TripletSemiHardLoss(margin=hps.model.margin)
    else:
        raise NotImplementedError
    model.compile(
        optimizer=optimizer,
        loss=loss_fn
    )

    # get data
    def parse_fn(example_proto):
        features = {"data": tf.io.FixedLenFeature((), tf.string),
                    "label": tf.io.FixedLenFeature((), tf.int64),
                    }
        parsed_features = tf.io.parse_single_example(example_proto, features)
        data = tf.io.decode_raw(parsed_features["data"], tf.float32)
        data = tf.reshape(data, shape=(hps.model.input_shape))
        return data, parsed_features["label"]

    def triplet_generator(filenames):
        def gen():
            for _ in range(hps.train.epochs*2):
                random.shuffle(filenames)
                dataset = tf.data.TFRecordDataset(filenames)
                dataset = dataset.map(parse_fn, num_parallel_calls=4)
                for example in dataset:
                    yield example
        return gen

    AUTOTUNE = tf.data.experimental.AUTOTUNE
    train_files = [os.path.join(hps.data.train_path, filename) for filename in sorted(os.listdir(hps.data.train_path))]
    train_dataset = tf.data.Dataset.from_generator(triplet_generator(train_files),
                                                   output_types=(tf.float32, tf.int32)).batch(hps.train.batch_size)
    train_dataset = train_dataset.prefetch(AUTOTUNE)

    dev_dataset = load_tfrecord(dirname=hps.data.dev_path)
    dev_dataset = dev_dataset.map(parse_fn, num_parallel_calls=4)
    dev_dataset = dev_dataset.batch(hps.train.batch_size)
    dev_dataset = dev_dataset.shuffle(hps.train.buffer_size)
    dev_dataset = dev_dataset.prefetch(AUTOTUNE)

    test_dataset = load_tfrecord(dirname=hps.data.test_path)
    test_dataset = test_dataset.map(parse_fn, num_parallel_calls=4)
    test_dataset = test_dataset.batch(hps.train.batch_size)
    test_dataset = test_dataset.shuffle(hps.train.buffer_size)
    test_dataset = test_dataset.prefetch(AUTOTUNE)

    for (x, y) in train_dataset:
        print("Example batch:", x.shape, y.shape, x, y)
        break

    steps_per_epoch = len(train_files) * 15 // hps.train.batch_size
    # callbacks
    callbacks = [
        ModelCheckpoint(os.path.join(hps.model_dir, "checkpoints"),
                        verbose=1, save_freq='epoch', save_best_only=True),
        TensorBoard(log_dir=os.path.join(hps.model_dir, "summary")),
        EqualErrorRate(train_dataset, dev_dataset, steps_per_epoch, interval=5, logger=logger)
    ]

    # fit
    history = model.fit(train_dataset,
                        epochs=hps.train.epochs,
                        validation_data=dev_dataset,
                        steps_per_epoch=steps_per_epoch,
                        shuffle=False,
                        callbacks=callbacks)

    # evaluate
    model.evaluate(test_dataset)


if __name__ == "__main__":
    main()
