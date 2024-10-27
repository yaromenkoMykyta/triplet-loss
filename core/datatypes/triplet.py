import os
from typing import List

import numpy as np

from dataclasses import dataclass

from tensorflow.keras.applications.inception_v3 import preprocess_input
import tensorflow as tf


@dataclass
class Path:
    file_path: str
    file_name: str

    def __repr__(self):
        return os.path.join(self.file_path, self.file_name)


@dataclass
class TripletPath:
    anchor: Path
    positive: Path
    negative: Path


@dataclass
class Triplet:
    anchor: np.ndarray
    positive: np.ndarray
    negative: np.ndarray

    def preprocess(self):
        self.anchor = preprocess_input(self.anchor)
        self.positive = preprocess_input(self.positive)
        self.negative = preprocess_input(self.negative)


@dataclass
class TripletBatch:
    triplets: List[Triplet]

    def preprocess(self):
        for triplet in self.triplets:
            triplet.preprocess()

    def __repr__(self):
        return (
            np.asarray([triplet.anchor for triplet in self.triplets]),
            np.asarray([triplet.positive for triplet in self.triplets]),
            np.asarray([triplet.negative for triplet in self.triplets]),
        )

    def __iter__(self):
        return iter(self.__repr__())


@dataclass
class TrainingConfig:
    margin: float
    epochs: int
    batch_size: int
    name: str
    save_path: str
    max_images_num_per_category: int
    max_images_num_per_test_category: int
    weights_format: str
    weights_model: str
    pooling: str
    initial_learning_rate: float
    first_decay_steps: int
    t_mul: float
    m_mul: float
    alpha: float
    pretrained_trainable_layers: int
    image_input_size: int
    split_value_for_dataset: float
