"""Data owner module

Use-case dependent logic should be implemented here in
"""

import logging
import os
import subprocess
import time
from abc import ABC, abstractmethod
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from typeguard import typechecked

import rpyc
from kafka import KafkaConsumer, KafkaProducer
from rpyc.utils.server import ThreadedServer
from torch.utils.data import DataLoader
import torch.nn.functional
import utilities
from config import Config
from saferlearn import Dataset, MPCParameters, SecureCollaborativeLearning

logging.getLogger("requests").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("kafka").setLevel(logging.WARNING)
logger: logging.Logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
local_config = Config()

# FIXME: change the way of setting ORCHESTRATOR variable to have something available in a distributed fashion
ORCHESTRATOR_HOST: Optional[str] = os.environ.get("ORCHESTRATOR_HOST", "192.168.12.26")
# FIXME : Directory path where model should be (TO CHANGE)
MODELS_PATH: Path = Path(local_config.MODELS_PATH, "")


class State(Enum):
    WAITING = 1
    RUNNING = 2
    DONE = 3


class DataOwner(rpyc.Service, ABC):
    _state: State = State.WAITING
    _dataset: Dataset
    _producer: KafkaProducer
    _id: int
    _tag: str
    _mpc_parameters: MPCParameters
    _worker_uuid: str
    _nb_teachers: int
    _protocol: SecureCollaborativeLearning

    def __init__(
        self,
        worker_uuid: str,
        nb_teachers: int,
        tag: str,
        mpc_parameters: MPCParameters,
        rpyc_host: str,
        rpyc_port: int,
    ) -> None:
        self._worker_uuid: str = worker_uuid
        self._nb_teachers: int = nb_teachers
        self._tag = tag
        self._mpc_parameters = mpc_parameters
        self._rpyc_host: str = rpyc_host
        self._rpyc_port: int = rpyc_port

    def exposed_get_state(self) -> str:
        """Returns the current state of the data owner
        Returns:
            str: The state (waiting, running, done)
        """
        return self._state.name

    def select_available_model(self, home_path: Path) -> Tuple[bool, Path]:
        """_summary_

        Args:
            home_path (Path): _description_

        Returns:
            Tuple[bool, Path]: _description_
        """
        if not home_path.exists():
            logging.error("Invalid path : %s", home_path)
            return (False, Path())
        models: List[Path] = [model for model in home_path.iterdir() if model.is_dir()]
        for model in sorted(models, key=lambda x: int(x.stem)):
            lock_file_name: Path = model / ".lock"
            if lock_file_name.exists():
                logging.info("%s directory is locked.", model)
            else:
                files: List[Path] = [file for file in model.iterdir() if file.is_file()]
                logging.debug(files)
                if len(files) > 0:
                    first_file: Path = files[0]
                    # Path(lock_file_name).touch()  # Add a lock
                    logging.debug("Loading model: %s", first_file)
                    return (True, first_file)
                else:
                    logging.error("No model found in %s", model)
                    Path(lock_file_name).touch()  # Add a lock

        logger.debug("%s : No available model!", home_path)

        # Beware, Path() point to '.' by default
        return (False, Path())

    def label_dataset(self, number_of_predictions: int) -> List[int]:
        """Label a public dataset using a private model

        Args:
            number_of_predictions (int): The number of sample to classify

        Returns:
            List[int]: The list of labels
        """
        # Retrieve local model
        model_selected, model_fname = self.select_available_model(MODELS_PATH)

        if not model_selected:
            raise FileNotFoundError("No free model found in ", MODELS_PATH)
        model = self.load_model(model_fname)

        logger.debug(model)

        # Retrieve public dataset
        logger.debug(f"Dataset : {self._dataset}")
        public_training_dataloader = self.load_public_training_dataset(
            self._dataset.name
        )

        # inference
        predictions: List[int] = self.infer_label(model, public_training_dataloader)
        logger.debug("Dataset : %s", public_training_dataloader)
        # logger.debug("Predictions : %s", predictions)
        # logger.debug("True labels : %s", truth)

        return predictions

    def exposed_label_dataset(
        self,
        dataset: str,
        nb_classes: int,
        nb_teachers: int,
        job_uuid: str,
        encrypt: bool,
        number_of_predictions: int,
        differential_privacy_noise_parameter: float,
    ):
        if encrypt:
            return self.exposed_label_dataset_fhe(
                Dataset(dataset, "tmp", nb_classes),
                job_uuid,
                nb_teachers,
                number_of_predictions,
            )
            logger.debug(f"Dataset loaded: {dataset}")
        else:
            self._dataset = Dataset(dataset, "tmp", nb_classes)
            self._nb_teachers = nb_teachers
            predictions: List[int] = self.label_dataset(number_of_predictions)
            self.send_encrypted_labels(predictions, job_uuid, encrypt=encrypt)

        logger.debug(f"Nb prediction {len(predictions)}")

        # logger.info("Calling finished_job")
        # if ORCHESTRATOR_HOST is not None and self._rpyc_port is not None:
        #     utilities.register_finish(
        #         ORCHESTRATOR_HOST,
        #         "5000",
        #         self._worker_uuid,
        #         job_uuid,
        #         self._rpyc_host,
        #         self._rpyc_port,
        #     )

        return len(predictions)

    def exposed_label_dataset_fhe(
        self,
        dataset: Dataset,
        job_uuid: str,
        nb_teachers=3,
        number_of_predictions=30,
    ) -> None:
        """Exposed method to label a public dataset by a PATE teacher and send the
        encrypted labels to a homomorphic aggregator

        Args:
            dataset (Dataset): _description_
            job_uuid (str): _description_
            protocol (SecureCollaborativeLearning): _description_
            hosts (List[Dict[str, str]]): _description_
            client_id (int, optional): _description_. Defaults to -1.
            last_flag (int, optional): _description_. Defaults to 0.
            number_of_predictions (int, optional): _description_. Defaults to 30.
        """
        self._dataset = dataset
        self._nb_teachers = nb_teachers
        predictions: List[int] = self.label_dataset(number_of_predictions)
        self.send_encrypted_labels(predictions, job_uuid)
        return len(predictions)

    def exposed_label_dataset_mpc(
        self,
        dataset: Dataset,
        job_uuid: str,
        hosts: List[Dict[str, str]],
        nb_teachers=3,
        client_id=-1,
        last_flag=0,
        number_of_predictions=30,
    ) -> None:
        """Exposed method to label a public dataset by a PATE teacher and send the
        labels to MPC aggregators

        Args:
            dataset (Dataset): _description_
            job_uuid (str): _description_
            hosts (List[Dict[str, str]]): _description_
            nb_teachers (int, optional): _description_. Defaults to 3.
            client_id (int, optional): _description_. Defaults to -1.
            last_flag (int, optional): _description_. Defaults to 0.
            number_of_predictions (int, optional): _description_. Defaults to 30.
        """
        self._dataset = dataset
        self._nb_teachers = nb_teachers
        self._mpc_parameters.set_hosts(hosts)
        self._mpc_parameters.set_flag(last_flag)
        predictions: List[int] = self.label_dataset(number_of_predictions)
        self.aggregate_labels_mpc(predictions, client_id, job_uuid)
        return len(predictions)

    def send_encrypted_labels(
        self, predictions: List[int], job_uuid: str, encrypt=True
    ) -> None:
        """_summary_

        Args:
            predictions (List[int]): _description_
            job_uuid (str): _description_
        """
        encode = False
        final_data: List
        topic_name = "vote_he"
        if not encrypt:
            encode = True
            final_data = predictions
            topic_name = "votes_pate"
        else:
            logger.info("Encrypting the labels")
            teacher_data_directory: Path = Path("/opt/pate_he/teacher/data")
            fhe_encryption_executable: Path = Path("/opt/pate_he/teacher/bin/encrypt")
            he_votes_bytes: List[bytes] = self.encrypt_votes(
                predictions,
                self._dataset.nb_classes,
                self._rpyc_host,
                teacher_data_directory,
                fhe_encryption_executable,
            )
            final_data = he_votes_bytes

        logger.info("Sending the labels")
        # logger.debug(f"Need to send {predictions}")
        self._producer: KafkaProducer = utilities.connect_kafka_producer()
        # Worker has been previously defined to RPYC_HOST to map to the IP address.
        # Changing to UUID in order to be more generic
        from alive_progress import alive_bar

        with alive_bar(len(final_data), bar="bubbles") as bar:
            for sample, he_vote_bytes in enumerate(final_data):
                utilities.publish_message(
                    self._producer,
                    topic_name,
                    '{"key": "'
                    + str(sample)
                    + '", "worker": "'
                    + self._worker_uuid
                    + '"}',
                    he_vote_bytes,
                    encode=encode,
                )
                bar()
            # utilities.publish_message(
            #     self._producer, f"nb_votes-{self._worker_uuid}", "number_votes", len(final_data)
            # )

        if self._producer is not None:
            self._producer.flush()
            self._producer.close()
        return

    def encrypt_votes(
        self,
        selected_class_indices: List[int],
        number_of_classes: int,
        teacher_id: str,
        teacher_data_directory: Path,
        fhe_encryption_executable: Path,
    ) -> List[bytes]:
        """_summary_

        Args:
            selected_class_indices (List[int]): _description_
            number_of_classes (int): _description_
            teacher_id (str):  Teacher's identification (IP)
            teacher_data_directory (Path): _description_
            fhe_encryption_executable (Path): _description_

        Returns:
            List[bytes]: _description_
        """
        vote_bits_array_enc: List[bytes] = []

        logger.debug(selected_class_indices)

        plain_vote_folder: Path = (
            teacher_data_directory / f"teacher_votes/plain_vote_{teacher_id}"
        )
        encrypted_vote_folder: Path = (
            teacher_data_directory / f"ciphertexts/encrypted_vote_{teacher_id}"
        )
        try:
            plain_vote_folder.mkdir(parents=True, exist_ok=True)
            encrypted_vote_folder.mkdir(parents=True, exist_ok=True)
        except OSError as error:
            logger.error("Error while creating directory: %s", error)

        logger.debug("Writing votes in file")
        total_start: float = time.time()
        for vote_position, class_index in enumerate(selected_class_indices):
            logger.debug("Vote #%s over %s", vote_position, len(selected_class_indices))
            if class_index >= number_of_classes:
                logger.error(
                    "Prediction is outside the allowed classes: %s out of %s classes",
                    class_index,
                    number_of_classes,
                )
            vote_bits_array_plain: List[int] = [0] * number_of_classes
            vote_bits_array_plain[class_index] = 1
            logger.debug(vote_bits_array_plain)
            with open(
                plain_vote_folder / str(vote_position),
                "w",
                encoding="utf-8",
            ) as plain_vote_file:
                for vote in vote_bits_array_plain:
                    plain_vote_file.write(str(vote) + " ")
            logger.debug("Encrypting array")
            try:
                encryption_subprocess: subprocess.CompletedProcess = subprocess.run(
                    [
                        fhe_encryption_executable,
                        teacher_data_directory / "keys/pub_params",
                        teacher_data_directory / "keys/pub_key",
                        str(self._nb_teachers),
                        str(self._dataset.nb_classes),
                        plain_vote_folder / str(vote_position),
                        encrypted_vote_folder / str(vote_position),
                    ],
                    capture_output=True,
                    check=True,
                )
                utilities.log_subprocess(encryption_subprocess)
            except subprocess.CalledProcessError as process_exception:
                utilities.log_subprocess(process_exception)
            with open(
                encrypted_vote_folder / str(vote_position),
                "rb",
            ) as encrypted_vote_file:
                vote_bits_array_enc.append(encrypted_vote_file.read())
        total_end: float = time.time()
        logger.debug("Total Encryption took %ss", total_end - total_start)

        return vote_bits_array_enc

    def aggregate_labels_mpc(self, predictions, client_id: int, job_uuid: str) -> None:
        """_summary_

        Args:
            predictions (_type_): _description_
            client_id (int): _description_
            job_uuid (str): _description_
        """
        self.write_mpc_votes(predictions, client_id)
        # rewrite HOSTS file
        with open("/opt/HOSTS", "w", encoding="utf-8") as hosts_file:
            for host in self._mpc_parameters.mpc_hosts:
                hosts_file.write(str(host["ip"]) + "\n")

        executable: Path = Path("/MPC") / f"{self._mpc_parameters.program}.x"

        if self._tag == "privacy_guardian":
            try:
                mpc_subprocess: subprocess.CompletedProcess = subprocess.run(
                    [
                        executable,
                        str(self._mpc_parameters.number_of_computing_parties),
                        str(self._mpc_parameters.size_of_batches),
                        str(self._nb_teachers),
                        "/opt/HOSTS",
                        str(self._mpc_parameters.sigma),
                        str(self._mpc_parameters.rounds),
                    ],
                    capture_output=True,
                    check=True,
                )
                utilities.log_subprocess(mpc_subprocess)
            except subprocess.CalledProcessError as process_exception:
                utilities.log_subprocess(process_exception)
        else:
            self._state = State.RUNNING
            start: float = time.time()
            try:
                mpc_subprocess = subprocess.run(
                    [
                        executable,
                        str(client_id),
                        str(self._mpc_parameters.number_of_computing_parties),
                        str(self._mpc_parameters.last_flag),
                        "/opt/HOSTS",
                        str(self._mpc_parameters.size_of_batches),
                        self._dataset.path,
                    ],
                    capture_output=True,
                    check=True,
                )
                if client_id == "0":
                    logger.info("Computation time : %s", time.time() - start)
                self._state = State.DONE
                utilities.log_subprocess(mpc_subprocess)
            except subprocess.CalledProcessError as process_exception:
                utilities.log_subprocess(process_exception)
        logger.info("Calling finished_job")
        if ORCHESTRATOR_HOST is not None and self._rpyc_port is not None:
            utilities.register_finish(
                ORCHESTRATOR_HOST,
                "5000",
                self._worker_uuid,
                job_uuid,
                self._rpyc_host,
                self._rpyc_port,
            )
        if client_id == "0":
            # MPC done, publish on kafka topic
            logger.info("Cleaning Topic")
            utilities.clean_topic("model_mpc")
            self._producer = utilities.connect_kafka_producer()
            logger.info("Publishing result MPC")
            with open("/opt/result.file", "r", encoding="utf-8") as result_file:
                vote_id: int = 0
                for line in result_file:
                    utilities.publish_message(
                        self._producer, "model_mpc", vote_id, line
                    )
                    vote_id += 1
        if self._producer is not None:
            self._producer.close()

    def write_mpc_votes(self, votes: List[int], client_id: int) -> None:
        """_summary_

        Args:
            votes (List[int]): _description_
            client_id (int): _description_
        """
        logger.debug("Writing votes in file for MPC")
        dataset_dir: Path = Path("/opt/input-data/" + self._dataset.name)
        try:
            dataset_dir.mkdir(parents=True, exist_ok=True)
        except OSError as error:
            logger.error("Error while creating directory: %s", error)
        with open(
            dataset_dir / str(client_id), "w", encoding="utf-8"
        ) as plaintext_file:
            for vote in votes:
                plaintext_file.write(str(vote) + "\n")

    def receive_keys(self) -> None:
        """Receive HE keys through Kafka"""
        topic_name: str = "keys"
        key_dir: Path = Path("/opt/pate_he/teacher/data/keys/")
        kafka_consumer: KafkaConsumer = utilities.connect_kafka_consumer(
            topic_name, group_id=self._worker_uuid
        )

        utilities.receive_file_kafka(kafka_consumer, key_dir, "pub_key")
        utilities.receive_file_kafka(kafka_consumer, key_dir, "pub_params")

    def launch_server(self) -> None:
        """_summary_"""
        server: ThreadedServer = ThreadedServer(
            self,
            hostname="0.0.0.0",
            port=int(self._rpyc_port),
            auto_register=False,
            protocol_config={
                "instantiate_custom_exceptions": True,
                "import_custom_exceptions": True,
            },
            backlog=500,
        )
        server.start()

    # Abstract methods to implement to add a use-case
    @typechecked
    @abstractmethod
    def load_public_training_dataset(self, dataset: str) -> DataLoader:
        """Load the public training samples

        Args:
            dataset (str): _description_

        Returns:
            DataLoader: _description_
        """

    @typechecked
    @abstractmethod
    def load_model(self, model_name: str) -> torch.nn.functional:
        """Load the model

        Args:
            model_name (str): Name of the model from torchvision
        """

    @typechecked
    @abstractmethod
    def infer_label(
        self,
        model: Any,
        dataset: Any,
        number_of_predictions=30,
    ) -> List[int | str]:
        """Do some Machine learning prediction stuff and return an array of prediction

        Args:
            model (torch.nn.Sequential): _description_
            dataset (Variable): _description_
            slot_count (int, optional): Size of the matrix. Defaults to 4096.
            row_size (int, optional): Slot_count divided by 2. Defaults to 2048.

        Returns:
            List[int | str]: _description_
        """
