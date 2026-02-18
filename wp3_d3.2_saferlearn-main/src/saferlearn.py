from enum import Enum
from pathlib import Path
from typing import Dict, List


class SecureCollaborativeLearning(Enum):
    PATE = "PATE"
    PATE_MPC = "PATE MPC"
    PATE_FHE = "PATE HE"
    FEDERATED_LEARNING = "Federated Learning"


class Dataset:
    name: str
    path: Path
    nb_classes: int

    def __init__(self, name: str, path: Path, nb_classes: int) -> None:
        self.name = name
        self.path = path
        self.nb_classes = nb_classes


class MPCParameters:
    number_of_computing_parties: int
    size_of_batches: int
    program: str
    sigma: int
    rounds: int
    mpc_hosts: List[Dict[str, str]]
    last_flag: int

    def __init__(
        self,
        number_of_computing_parties: int,
        size_of_batches: int,
        program: str,
        sigma: int = 1,
        rounds: int = -1,
    ) -> None:
        self.number_of_computing_parties = number_of_computing_parties
        self.size_of_batches = size_of_batches
        self.program = program
        self.sigma = sigma
        self.rounds = rounds

    def set_hosts(self, hosts: List[Dict[str, str]]) -> None:
        """Setter for the list of aggregators hosts

        Args:
            hosts (List[Dict[str, str]]): The list of aggregators
        """
        self.mpc_hosts = hosts

    def set_flag(self, flag: int) -> None:
        """Setter for the flag indicating the last MPC client

        Args:
            flag (int): The value (1 if last client)
        """
        self.last_flag = flag
