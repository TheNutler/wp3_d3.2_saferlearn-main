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


class deeplog(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_keys):
        super(deeplog, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_keys)

    def forward(self, features, device):
        input0 = features[0]
        h0 = torch.zeros(self.num_layers, input0.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, input0.size(0), self.hidden_size).to(device)
        out, _ = self.lstm(input0, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out


class AITDataOwner(DataOwner):
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
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
        model = deeplog().to(device)
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
        # TODO

        return

    @typechecked
    def load_model(self, model_path: Path) -> torch.nn.functional:
        """Load the model

        Args:
            model_path (str): Path to the model stored on disk

        Returns:
        torch.nn.functionnial, the model ready to perform the prediction
        Note: the model class is needed to load the model
        """
        model = deeplog()
        logging.debug("Trying to load %s", model_path)
        with open(model_path, "rb") as model_file:
            model.load_state_dict(torch.load(model_file))
        logging.debug("Loaded this model: %s", model)

        return model
