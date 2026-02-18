#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File              : UC_stub.py
# Author            : Alice Héliou <alice.heliou@thalesgroup.com>
# Date              : 02.03.2023
# Last Modified Date: 02.03.2023
# Last Modified By  : Alice Héliou <alice.heliou@thalesgroup.com>

import logging
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import (
    Conv1D,
    Input,
)
from tensorflow.keras.models import Model

logging.basicConfig(level=logging.DEBUG)


class UCStubModel(nn.Module):
    # Torch model used for the Pate local model
    # The model definition is needed to load the model
    def __init__(self, shape, n_classes):
        super().__init__()

    def forward(self, x):
        return x


class PateUtils:
    def __init__(
        self,
        public_dataset_path: str = "/data/public-dataset/my_public_data",
        model_test_path: str = "/tmp/trained_nets/trained_net/my_trained_net.pkl",
        models_path: str = "/tmp/trained_nets/",
        input_shape: Tuple[int] = (1024, 2),
    ):
        self.public_dataset_path = public_dataset_path
        self.model_test_path = model_test_path
        self.models_path = models_path
        self.input_shape = input_shape

    def get_dataset(self, dataset_path: str, number_of_samples: int) -> np.ndarray:
        """Get partitioned dataset

        Args:
            dataset (str): Path to the dataset

        Returns:
        np.ndarray
        We do not partion the data, but directly provide a file containing the data to label.
        """
        logging.debug(f"Try to read {dataset_path}")
        dataset_path = self.public_dataset_path
        logging.debug(f"Try to read {dataset_path}")
        data = np.genfromtxt(dataset_path, delimiter=",")
        logging.debug("Data readen")
        one_data_size = 1
        for d in self.input_shape:
            one_data_size *= d
        signals = np.reshape(
            data[:, :one_data_size],
            (data.shape[0], self.input_shape[0], self.input_shape[1]),
        )
        class_onehot = data[:, one_data_size:]
        print(np.argmax(class_onehot[:30], -1))

        return signals[:number_of_samples], class_onehot[:number_of_samples]

    def get_model(self, model_path: str) -> torch.nn.Sequential:
        """Load the model

        Args:
            model_path (str): Path to the model stored on disk

        Returns:
        torch.nn.Sequential, the model ready to perform the prediction
        Note: the model class is needed to load the model
        """
        model = None
        logging.debug(f"Trying to load {model_path}")
        with open(model_path, "rb") as model_file:
            model: torch.nn.Sequential = torch.load(model_file).double()
        logging.debug(model)
        return model

    def do_prediction(
        self,
        model: torch.nn.Sequential,
        data: np.ndarray,
    ) -> List[int]:
        """Do some Machine learning prediction stuff and return an array of prediction
        8
                Args:
                    model (torch.nn.Sequential): model to perform the prediction
                    dataset (np.ndarray): dataset, one row per data
                Returns:
                    List[int]: List of prediction, one integer per data
        """
        logging.debug("Doing prediction....")
        if model is None:
            logging.debug("model is None")
            return []
        torch_data = torch.from_numpy(data.transpose(0, 2, 1)).double()
        predictions = model(torch_data)
        result: torch.Tensor = torch.argmax(predictions, -1)
        logging.debug(result.shape)
        logging.debug("Prediction done....")
        return list(result.numpy())

    def test(self):
        """Methods to test the other methods
        It reads the public_dataset_path, and the model_test_path.
        It make the prediction and output them

        Args:
            model_path (str) : path to the model to perform the prediction
            dataset_path (str) : path to the dataset, one row per data

        """
        data, truth = self.get_dataset(self.public_dataset_path)
        model = self.get_model(self.model_test_path)
        predictions: List[int] = self.do_prediction(model, data)
        logging.debug("%s elements : %s", len(predictions), predictions[:30])


class FlUtils:
    # Methods and parameters concerning the remote dataset and the model to train

    def __init__(
        self,
        input_shape: Tuple[int] = (1024, 2),
        n_classes: int = 4,
        batch_size: int = 512,
        nb_epochs: int = 10,
        test_datapath: int = "/data/public-dataset/my-data-test.csv",
        offset: int = 0,
        header: int = False,
        index: int = False,
    ):
        self.input_shape = input_shape
        self.n_classes = n_classes

        self.batch_size = batch_size
        self.nb_epochs = nb_epochs
        self.test_datapath = test_datapath
        self.offset = offset
        self.header = header
        self.index = index
        self.X_test, self.y_test = self.read_csv(self.test_datapath)

    def get_remote_data_paths(self, nb_participants: int):
        """
        A data method, that stores the local datapath of the participants

        """
        remote_participant_data_paths = [
            "/Data/0/participant_0.csv",
            "/Data/1/participant_1.csv",
            "/Data/2/participant_2.csv",
        ][:nb_participants]
        return remote_participant_data_paths

    def read_csv(self, filepath: str) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Method to read the dataset and preproces it

        Args:
            filepath (str) : path the data file to read

        Returns;
            [X,y]: a Tuple a tf.Tensor,
                X the train_data preprocessed to be ready to feed the model
                y the labels of the train_data

        """
        _ = pd.read_csv(filepath, sep=",", header=None)
        X = None
        y = None

        return X, y

    def make_model(self) -> tf.keras.models.Model:
        """Define a Keras model without much of regularization
        Such a model is prone to overfitting"""

        i = Input(shape=self.input_shape)

        # FIXME: dummy model to be implemented in relation to federated learning
        x = Conv1D()(i)

        model = Model(i, x)
        return model

    @staticmethod
    def myloss_fn():
        return tf.keras.losses.CategoricalCrossentropy()

    @staticmethod
    def mymetric_fn():
        return tf.keras.metrics.CategoricalAccuracy()

    @staticmethod
    def myopt():
        return tf.keras.optimizers.Adam()


if __name__ == "__main__":
    Pate_test = PateUtils()
    Pate_test.test()
