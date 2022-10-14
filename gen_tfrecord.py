import tensorflow as tf
from tqdm import tqdm
import numpy as np
from util import feature_extractor
from util.dataio import save_tfrecord
import json
import os


def gen(label_idx=0, n_classes=50000, scenario="train", norm=None, data_dir=None):

    saved_dir = os.path.join(data_dir, '{}_{}'.format(norm, n_classes - label_idx), scenario)
    os.makedirs(saved_dir)

    min_sample = 10
    max_len = 70
    pad_value = 0.
    error_rate = 0.2

    with open("/media/hoang/Data/keystroke_dataset/Keystrokes/features/all_by_user.json") as f:
        for idx, line in tqdm(enumerate(f)):

            # start from label_idx
            if idx < label_idx:
                continue

            # each user
            line = json.loads(line)
            if len(line['sequences']) < min_sample:
                continue

            X = []
            Y = []

            # each sequence of user
            for data in line['sequences']:
                # flatten to raw_data
                raw_data = []
                raw_text = None
                for d in data:
                    new_d = {'time': d['press_time'], 'keycode': d['keycode'], 'type': 'down'}
                    raw_data.append(new_d)
                    new_d = {'time': d['release_time'], 'keycode': d['keycode'], 'type': 'up'}
                    raw_data.append(new_d)
                    if raw_text is None:
                        raw_text = d['text']

                # compare raw data/text
                raw_data = sorted(raw_data, key=lambda x: x['time'])
                if len(raw_text)*2*(1 - error_rate) <= len(raw_data) <= len(raw_text)*2*(1 + error_rate):
                    feature, duration = feature_extractor.extract(raw_data)
                    x = feature_extractor.input_from_feature(feature, duration, norm=norm)
                    # append
                    if x is None or np.isnan(x).any():
                        continue
                    X.append(x)
                    Y.append(label_idx)

            X = tf.keras.preprocessing.sequence.pad_sequences(X,
                                                              padding="pre",
                                                              value=pad_value,
                                                              maxlen=max_len,
                                                              dtype="float")
            # concat
            X = np.array(X)
            Y = np.array(Y)

            if len(X) >= min_sample:
                # save dataset
                save_tfrecord(data=X, label=Y, filepath=os.path.join(saved_dir, "{}.tfrecord".format(label_idx)))
                label_idx += 1
                if label_idx == n_classes:
                    print("DONE")
                    return


if __name__ == '__main__':
    norm = 'none'
    data_dir = "data/"
    # train
    gen(label_idx=0, n_classes=1000, scenario="train", norm=norm, data_dir=data_dir)
    # dev
    # gen(label_idx=128000, n_classes=128000+10000, scenario="dev", norm=norm, data_dir=data_dir)
    # # test
    # gen(label_idx=148000, n_classes=148000+10000, scenario="test", norm=norm, data_dir=data_dir)

