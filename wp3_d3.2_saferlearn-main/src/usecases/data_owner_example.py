import logging as logger
from pathlib import Path
import torch.utils.data
from typing import List

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

import torchvision
from typeguard import typechecked

from usecases.data_owner_abstract_class import DataOwner

logger.basicConfig(level=logger.DEBUG)
logger.getLogger("requests").setLevel(logger.WARNING)
logger.getLogger("urllib3").setLevel(logger.WARNING)
logger.getLogger("kafka").setLevel(logger.WARNING)
logging: logger.Logger = logger.getLogger(__name__)


class UCStubModel(nn.Module):
    """
    A class used to represent a stub model for use cases.

    ...

    Attributes
    ----------
    conv1 : nn.Conv2d
        a convolutional layer with 1 input image channel, 32 output channels, and a square 3x3 convolution
    conv2 : nn.Conv2d
        a convolutional layer with 32 input image channel, 64 output channels, and a square 3x3 convolution
    dropout1 : nn.Dropout
        a dropout layer with p=0.25
    dropout2 : nn.Dropout
        a dropout layer with p=0.5
    fc1 : nn.Linear
        a linear layer with 9216 input features and 128 output features
    fc2 : nn.Linear
        a linear layer with 128 input features and 10 output features

    Methods
    -------
    forward(x)
        Defines the computation performed at every call.
    """

    def __init__(self):
        super(UCStubModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


class ThalesDataOwner(DataOwner):
    """Thales radio's use-case data owner"""

    def infer_label(
        self,
        model: torch.nn.Sequential,
        dataset: torch.utils.data.DataLoader,
    ) -> List[int]:
        """Do some Machine learning prediction stuff and return an array of prediction

        Args:
            model (torch.nn.Sequential): _description_
            dataset (Variable): _description_

        Returns:
            List[int]: _description_
        """
        if torch.cuda.is_available():
            device = torch.device("cuda:1")
        else:
            device = torch.device("cpu")
        model = UCStubModel().to(device)
        train_kwargs = {}
        if torch.cuda.is_available():
            cuda_kwargs = {
                "num_workers": 1,
                "pin_memory": True,
                "shuffle": True,
                "batch_size": 1024,
            }
            train_kwargs.update(cuda_kwargs)
        logging.debug("Doing prediction....")
        result_np: np.array = np.array([int])
        for data, target in dataset:
            data = data.to(device)
            outputs = model(data)
            _, result = torch.max(outputs, 1)
            for pred in result:
                result_np = np.append(result_np, pred.cpu().detach().numpy())

        logging.debug(f"Nb of predictions {len(result_np)}")
        logging.debug(f"Predictions {result_np}")
        logging.debug("Prediction done....")

        return list(result_np.tolist()[1:])

    def load_public_training_dataset(
        self, dataset_name: str
    ) -> torch.utils.data.DataLoader:
        """Load and partition the public dataset

        We do not partition the data, but directly provide a file containing the data to label.

        Args:
            dataset_path (str): Path to the dataset

        Returns:
            Tuple[np.ndarray, np.ndarray]: _description_
        """
        train_kwargs = {}
        if torch.cuda.is_available():
            cuda_kwargs = {
                "num_workers": 1,
                "pin_memory": True,
                "shuffle": True,
                "batch_size": 1024,
            }
            train_kwargs.update(cuda_kwargs)
        transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Grayscale(num_output_channels=1),
                torchvision.transforms.Normalize((0.5,), (0.5,)),
                # torchvision.transforms.Normalize((0.1307,), (0.3081,)),
                # torchvision.transforms.Resize((32, 32)),
            ]
        )

        dataset = getattr(torchvision.datasets, dataset_name)(
            "/tmp/data", train=False, download=True, transform=transform
        )

        return torch.utils.data.DataLoader(dataset, **train_kwargs)

    @typechecked
    def load_model(self, model_path: Path) -> torch.nn.functional:
        """Load the model

        Args:
            model_path (str): Path to the model stored on disk

        Returns:
        torch.nn.functionnial, the model ready to perform the prediction
        Note: the model class is needed to load the model
        """
        model = UCStubModel()
        logging.debug("Trying to load %s", model_path)
        with open(model_path, "rb") as model_file:
            model.load_state_dict(torch.load(model_file))
        logging.debug("Loaded this model: %s", model)
        return model
