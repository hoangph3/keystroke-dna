from tensorflow.keras import callbacks
from .metric import equal_error_rate
from tqdm import tqdm
import numpy as np


class EqualErrorRate(callbacks.Callback):
    def __init__(self, train_dataset, validation_dataset, train_steps=None, validation_steps=None, interval=5, logger=None):
        super(EqualErrorRate, self).__init__()
        self.train_dataset = train_dataset
        self.validation_dataset = validation_dataset
        self.train_steps = train_steps
        self.validation_steps = validation_steps
        self.interval = interval
        self.logger = logger

    def compute_eer(self, dataset, n_steps=None, threshold=None):
        thresholds = []
        false_positive = []
        false_negative = []
        if n_steps is None:
            n_steps = sum(1 for _ in dataset)
        for i, batch in tqdm(enumerate(dataset), desc="Evaluate", total=n_steps):
            if i > n_steps:
                break
            x_batch, y_true_batch = batch
            y_pred_batch = self.model.predict(x_batch, verbose=0)
            if threshold is None:
                fp, fn, threshold = equal_error_rate(y_true_batch, y_pred_batch)
            else:
                fp, fn, threshold = equal_error_rate(y_true_batch, y_pred_batch, threshold)
            false_positive.append(fp)
            false_negative.append(fn)
            thresholds.append(threshold)
        return float(np.mean(false_positive)), float(np.mean(false_negative)), float(np.mean(thresholds))

    def on_epoch_end(self, epoch, logs=None):
        if epoch % self.interval == 0:
            false_positive, false_negative, threshold = self.compute_eer(self.train_dataset, self.train_steps)
            logs["threshold"] = threshold

            logs["false_positive"] = false_positive
            logs["false_negative"] = false_negative

            val_false_positive, val_false_negative, threshold = self.compute_eer(self.validation_dataset, threshold)
            logs["val_false_positive"] = val_false_positive
            logs["val_false_negative"] = val_false_negative

            self.logger.info(logs)
