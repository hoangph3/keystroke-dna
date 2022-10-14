import json

import numpy as np


class Feature:
    def __init__(
            self,
            vocab_path="configs/vocab.json",
            feature_type_path="configs/feature_type.json"
    ):
        with open(vocab_path) as f:
            self.vocab = json.load(f)

        with open(feature_type_path) as f:
            self.feature_vocab = json.load(f)

    def input_from_raw(self, raw_seq):
        features = self.extract(raw_seq)

        return self.input_from_feature(features)

    def input_from_feature(self, features):
        raise NotImplementedError

    def extract_key(self, sub_seq):
        raise NotImplementedError

    def agg_feature(self, feature, features=None):
        raise NotImplementedError

    def extract(self, raw_seq):
        raw_seq = Feature.clean_raw_seq(raw_seq)
        duration = (raw_seq[-1]['time'] - raw_seq[0]['time'])

        features = None
        for i in range(len(raw_seq)):
            features = self.agg_feature(self.extract_key(raw_seq[i:]), features)

        return features, duration

    @staticmethod
    def clean_raw_seq(data):
        data = sorted(data, key=lambda x: x["time"])

        i = 0
        while i < len(data):
            if "keycode" not in data[i]:
                data.pop(i)
                continue
            i += 1

        return data


class MatrixFeature(Feature):
    def input_from_feature(self, features):
        n_features = 5
        n_keycodes = len(self.vocab)
        feature_matrix = np.full((n_keycodes, n_keycodes, n_features), 0, dtype=np.float32)

        for key, value in features.items():
            key_items = key.split('_')
            source_key = key_items[0]
            feature_type = key_items[-1]
            target_key = source_key if feature_type == "Hold" else key_items[1]

            value = [item for item in value if (item < self.feature_vocab[feature_type]["max"]) and (item > 0)]

            if not value:
                continue

            if source_key not in self.vocab:
                continue

            if target_key not in self.vocab:
                continue

            value = np.array(value)
            feature_matrix[
                self.vocab[source_key], self.vocab[target_key], self.feature_vocab[feature_type]["index"]
            ] = np.mean(value)

        feature_matrix = feature_matrix / 1000.

        return feature_matrix

    def extract_key(self, sub_seq):
        features = dict()
        source_down = {}
        source_up = {}
        target_down = {}
        target_up = {}

        for step_idx, step in enumerate(sub_seq):
            if step["type"] == "down":
                if not source_down:
                    source_down = step
                    continue

                if not target_down:
                    target_down = step
                    continue

            if step["type"] == "up":
                if (not source_up) and source_down and (step["keycode"] == source_down["keycode"]):
                    source_up = step
                    continue

                if (not target_up) and target_down and (step["keycode"] == target_down["keycode"]):
                    target_up = step
                    continue

            if source_down and source_up and target_down and target_up:
                break

        if (not source_down) or (not source_up) or (not target_down) or (not target_up):
            return {}

        features["{}_{}_DD".format(
            source_down["keycode"],
            target_down["keycode"]
        )] = target_down["time"] - source_down["time"]

        features["{}_{}_DU".format(
            source_down["keycode"],
            target_up["keycode"]
        )] = target_up["time"] - source_down["time"]

        features["{}_{}_UD".format(
            source_up["keycode"],
            target_down["keycode"]
        )] = target_down["time"] - source_up["time"]

        features["{}_{}_UU".format(
            source_up["keycode"],
            target_up["keycode"]
        )] = target_up["time"] - source_up["time"]

        features["{}_Hold".format(
            source_down["keycode"]
        )] = source_up["time"] - source_down["time"]

        return features

    def agg_feature(self, feature, features=None):
        if not features:
            features = dict()

        for key, value in feature.items():
            features[key] = features.get(key, [])
            features[key].append(value)

        return features


class StatsFeature(Feature):
    def input_from_feature(self, features):
        n_features = 5
        n_keycodes = len(self.vocab)
        feature_mean = np.full((n_keycodes, n_features), 0, dtype=np.float32)
        feature_std = np.full((n_keycodes, n_features), 0, dtype=np.float32)

        for key, value in features.items():
            key_items = key.split('_')
            source_key = key_items[0]
            feature_type = key_items[-1]

            value = [item for item in value if (item < self.feature_vocab[feature_type]["max"]) and (item > 0)]

            if not value:
                continue

            if source_key not in self.vocab:
                continue

            value = np.array(value)
            feature_mean[self.vocab[source_key], self.feature_vocab[feature_type]["index"]] = np.mean(value)
            feature_std[self.vocab[source_key], self.feature_vocab[feature_type]["index"]] = np.std(value)

        feature_mean = feature_mean / 1000.
        feature_std = feature_std / 1000.

        return np.concatenate([feature_mean, feature_std], axis=-1)

    def extract_key(self, sub_seq):
        features = dict()
        source_down = {}
        source_up = {}
        target_down = {}
        target_up = {}

        for step_idx, step in enumerate(sub_seq):
            if step["type"] == "down":
                if not source_down:
                    source_down = step
                    continue

                if not target_down:
                    target_down = step
                    continue

            if step["type"] == "up":
                if (not source_up) and source_down and (step["keycode"] == source_down["keycode"]):
                    source_up = step
                    continue

                if (not target_up) and target_down and (step["keycode"] == target_down["keycode"]):
                    target_up = step
                    continue

            if source_down and source_up and target_down and target_up:
                break

        if (not source_down) or (not source_up) or (not target_down) or (not target_up):
            return {}

        features["{}_{}_DD".format(
            source_down["keycode"],
            target_down["keycode"]
        )] = target_down["time"] - source_down["time"]

        features["{}_{}_DU".format(
            source_down["keycode"],
            target_up["keycode"]
        )] = target_up["time"] - source_down["time"]

        features["{}_{}_UD".format(
            source_up["keycode"],
            target_down["keycode"]
        )] = target_down["time"] - source_up["time"]

        features["{}_{}_UU".format(
            source_up["keycode"],
            target_up["keycode"]
        )] = target_up["time"] - source_up["time"]

        features["{}_Hold".format(
            source_down["keycode"]
        )] = source_up["time"] - source_down["time"]

        return features

    def agg_feature(self, feature, features=None):
        if not features:
            features = dict()

        for key, value in feature.items():
            features[key] = features.get(key, [])
            features[key].append(value)

        return features


class AnonymousSeqFeature(Feature):
    def input_from_feature(self, features, duration, norm):
        steps = []

        for feature in features:
            step = [None for _ in feature]
            step[self.feature_vocab["DD"]["index"]] = feature["DD"]
            step[self.feature_vocab["DU"]["index"]] = feature["DU"]
            step[self.feature_vocab["UD"]["index"]] = feature["UD"]
            step[self.feature_vocab["UU"]["index"]] = feature["UU"]
            step[self.feature_vocab["Hold"]["index"]] = feature["Hold"]

            steps.append(step)

        steps = np.array(steps)
        if np.max(steps) > 1500:
            return

        # normalize
        # steps = steps * len(steps) / duration

        if norm == 'max':
            res = steps / np.max(steps)
        elif norm == 'min_max':
            res = (steps - np.min(steps)) / (np.max(steps) - np.min(steps))
        elif norm == 'none':
            res = steps / 1000.
        else:
            raise ValueError("Must norm")

        if np.isnan(res).any():
            print(steps)
        return res

    def extract_key(self, sub_seq):
        features = dict()
        source_down = {}
        source_up = {}
        target_down = {}
        target_up = {}

        for step_idx, step in enumerate(sub_seq):
            if step["type"] == "down":
                if not source_down:
                    source_down = step
                    continue

                if not target_down:
                    target_down = step
                    continue

            if step["type"] == "up":
                if step_idx == 0:
                    return {}

                if (not source_up) and source_down and (step["keycode"] == source_down["keycode"]):
                    source_up = step
                    continue

                if (not target_up) and target_down and (step["keycode"] == target_down["keycode"]):
                    target_up = step
                    continue

            if source_down and source_up and target_down and target_up:
                break

        if (not source_down) or (not source_up) or (not target_down) or (not target_up):
            return {}

        features["DD"] = target_down["time"] - source_down["time"]
        features["DU"] = target_up["time"] - source_down["time"]
        features["UD"] = target_down["time"] - source_up["time"]
        features["UU"] = target_up["time"] - source_up["time"]
        features["Hold"] = source_up["time"] - source_down["time"]

        return features

    def agg_feature(self, feature, features=None):
        if not features:
            features = list()

        if feature:
            features.append(feature)

        return features
