#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File              : data_owner.py
# Author            : hargathor <3949704+hargathor@users.noreply.github.com>
# Date              : 09.04.2019
# Last Modified Date: 29.04.2022
# Last Modified By  : hargathor <3949704+hargathor@users.noreply.github.com>

"""Data owner module"""

import argparse
import json
import logging
import math
import os
import socket
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import rpyc
import torch
from kafka import KafkaConsumer, KafkaProducer
from rpyc.utils import server
from torch import nn
from torch.autograd import Variable

import utilities

from usecases.UC_stub import UCStubModel
from usecases.UC_stub import PateUtils as UCStubUtils

logging.basicConfig(level=logging.DEBUG)
logging.getLogger("requests").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("kafka").setLevel(logging.WARNING)
logger: logging.Logger = logging.getLogger("data_owner")
logger.setLevel(logging.DEBUG)

RPYC_PORT: Optional[str] = os.environ.get("RPYC_PORT")
RPYC_HOST: str = socket.gethostbyname(socket.gethostname())
ORCHESTRATOR_HOST: Optional[str] = os.environ.get("ORCHESTRATOR_HOST")
ORCHESTRATOR_PORT: Optional[int] = os.environ.get("ORCHESTRATOR_PORT")


# FIXME : Directory path where model should be (TO CHANGE)
# MODELS_PATH: Path = Path("/tmp/trained_nets")


class Module_power(nn.Module):
    """Torch model"""

    def __init__(self, arg) -> None:
        super(Module_power, self).__init__()
        self.power = arg

    def forward(self, vector):
        """Torch shenanigans"""
        return (vector**self.power).mean(dim=2)


class WorkerService(rpyc.Service):
    """_summary_

    Args:
        rpyc (_type_): _description_

    Returns:
        _type_: _description_
    """

    _producer: Optional[KafkaProducer] = None
    state: str = "waiting"

    def __init__(
        self,
        worker_uuid: str,
        number_of_computing_parties: int,
        size_of_batches: int,
        program: str,
        tag: str,
        n_teachers: int,
        sigma: int,
        rounds: int,
    ) -> None:
        self.worker_uuid: str = worker_uuid
        self.number_of_computing_parties: int = number_of_computing_parties
        self.size_of_batches: int = size_of_batches
        self.program: str = program
        self.tag: str = tag
        self.nb_teachers: int = n_teachers
        self.sigma: int = sigma
        self.rounds: int = rounds
        self.nb_classes: int = 0
        self.usecase = None
        self.model_path = Path("/tmp/trained_nets")

    def exposed_get_state(self) -> str:
        """_summary_

        Returns:
            str: _description_
        """
        return self.state

    def exposed_get_predictions_mpc(
        self,
        dataset: str,
        dataset_path: str,
        my_hosts_str: str,
        # my_hosts: List[Dict[str, Any]],
        client_id: int,
        job_uuid: str,
        last_flag: int,
        number_of_predictions=1000,
    ) -> None:
        """_summary_

        Args:
            dataset (str): _description_
            dataset_path (str)
            my_hosts (List[Dict[str, str]]): _description_
            client_id (int): _description_
            job_uuid (str): _description_
            last_flag (int): _description_
            number_of_predictions (int, optional): _description_. Defaults to 1000.
        """
        logger.info("MPC aggregation")
        my_hosts = json.loads(my_hosts_str)
        # select the UC in function of the tag
        logger.info("****** Dataset selected %s  ******", dataset)
        if dataset == "StubDataset":
            self.usecase = UCStubUtils()

        self.models_path = Path(self.usecase.models_path)

        data, truth_one_hot = self.usecase.get_dataset(
            dataset_path, number_of_predictions
        )
        truth = [list(label).index(1) for label in truth_one_hot]

        logger.info("Looking for model in %s", self.models_path)
        model_loaded, model_fname = self.book_model_directory(self.models_path)
        if not model_loaded:
            raise FileNotFoundError("No free model found in ", self.models_path)
        logger.info("Loading model: %s", model_fname)
        model = self.usecase.get_model(model_fname)

        logger.info("Labeling the dataset")
        predictions: List[int] = self.usecase.do_prediction(model, data)

        logger.debug("Dataset : %s", data)
        logger.debug(predictions)
        logger.debug("Truth : %s", truth)

        self.remove_lock(model_fname)

        self.write_mpc_votes(predictions, dataset, client_id)
        # rewrite HOSTS file
        with open("/opt/HOSTS", "w", encoding="utf-8") as hosts_file:
            for host in my_hosts:
                hosts_file.write(str(host["ip"]) + "\n")

        executable: Path = Path("/MPC") / f"{self.program}.x"

        if self.tag == "privacy_guardian":
            try:
                mpc_subprocess: subprocess.CompletedProcess = subprocess.run(
                    [
                        executable,
                        str(self.number_of_computing_parties),
                        str(self.size_of_batches),
                        str(self.nb_teachers),
                        "/opt/HOSTS",
                        str(self.sigma),
                        str(self.rounds),
                    ],
                    capture_output=True,
                    check=True,
                )
                utilities.log_subprocess(mpc_subprocess)
            except subprocess.CalledProcessError as process_exception:
                utilities.log_subprocess(process_exception)
        else:
            self.state = "running"
            mpc_aggregation_start: float = time.time()
            try:
                mpc_subprocess = subprocess.run(
                    [
                        executable,
                        str(client_id),
                        str(self.number_of_computing_parties),
                        str(last_flag),
                        "/opt/HOSTS",
                        str(self.size_of_batches),
                        dataset,
                    ],
                    capture_output=True,
                    check=True,
                )
                if client_id == "0":
                    mpc_aggregation_end: float = time.time()
                    logger.debug(
                        "MPC aggregation took %ss",
                        mpc_aggregation_end - mpc_aggregation_start,
                    )
                self.state = "done"
                utilities.log_subprocess(mpc_subprocess)
            except subprocess.CalledProcessError as process_exception:
                utilities.log_subprocess(process_exception)
        logger.info("Calling finished_job")
        if ORCHESTRATOR_HOST is not None and RPYC_PORT is not None:
            utilities.register_finish(
                ORCHESTRATOR_HOST,
                ORCHESTRATOR_PORT,
                self.worker_uuid,
                job_uuid,
                RPYC_HOST,
                RPYC_PORT,
            )
        if client_id == "0":
            # MPC done, publish on kafka topic
            logger.info("Cleaning Topic")
            utilities.clean_topic("model_mpc")
            self._producer = utilities.connect_kafka_producer()
            logger.info("Publishing result MPC")
            with open(
                "/app_saferlearn/result.file", "r", encoding="utf-8"
            ) as result_file:
                vote_id: int = 0
                for line in result_file:
                    utilities.publish_message(
                        self._producer, "model_mpc", vote_id, line
                    )
                    vote_id += 1
        if self._producer is not None:
            self._producer.close()

    def book_model_directory(self, home_path: Path) -> Tuple[bool, Path]:
        """_summary_

        Args:
            home_path (Path): _description_

        Returns:
            Tuple[bool, Path]: _description_
        """
        if not home_path.exists():
            logger.error("Invalid path : %s", home_path)
            return (False, Path())
        models: List[Path] = [model for model in home_path.iterdir() if model.is_dir()]
        for model in sorted(models, key=lambda x: int(x.stem)):
            lock_file_name: Path = model / ".lock"
            if lock_file_name.exists():
                logger.info("%s directory is locked.", model)
            else:
                files: List[Path] = [file for file in model.iterdir() if file.is_file()]
                logger.debug(files)
                if len(files) > 0:
                    first_file: Path = files[0]
                    Path(lock_file_name).touch()  # Add a lock
                    logger.debug("Loading model: %s", first_file)
                    return (True, first_file)
                else:
                    logger.error("No model found in %s", model)
                    Path(lock_file_name).touch()  # Add a lock

        logger.debug("%s : No available model!", home_path)

        # Beware, Path() point to '.' by default
        return (False, Path())

    def remove_lock(self, model_path: Path):
        lock_file = model_path.parents[0] / ".lock"
        lock_file.unlink(missing_ok=True)
        return

    def receive_keys(self) -> None:
        """Receive HE keys through Kafka"""
        topic_name: str = "keys"
        key_dir: Path = Path("/app_saferlearn/pate_he/teacher/data/keys/")
        kafka_consumer: KafkaConsumer = utilities.connect_kafka_consumer(
            topic_name, group_id=self.worker_uuid
        )

        utilities.receive_file_kafka(kafka_consumer, key_dir, "pub_key")
        utilities.receive_file_kafka(kafka_consumer, key_dir, "pub_params")

    def exposed_label_dataset(
        self,
        dataset: str,
        nb_classes: int,
        nb_teachers: int,
        job_uuid: str,
        encrypt: bool,
        nb_samples: int,
        differential_privacy_noise_parameter: float,
    ) -> int:
        """_summary_

        Args:
            dataset (str): _description_
            nb_classes (int): _description_
            nb_teachers (int): _description_
            job_uuid (str): _description_
        """
        logger.info(f"****** Dataset selected {dataset}  ******")
        if dataset == "StubDataset":
            self.usecase = UCStubUtils()

        self.models_path = Path(self.usecase.models_path)

        # This is the starting point of the predictions
        predictions_start_time = time.time()

        predictions: List[int] = self.get_predictions(
            dataset, nb_classes, nb_teachers, nb_samples
        )

        # The predictions are over
        predictions_end_time = time.time()

        predictions_computation_time = predictions_end_time - predictions_start_time
        logger.info("Teacher predictions took %s s", predictions_computation_time)

        self.send_predictions(
            predictions, job_uuid, encrypt, differential_privacy_noise_parameter
        )
        return len(predictions)

    def get_predictions(
        self,
        dataset: str,
        nb_classes: int,
        nb_teachers: int,
        number_of_predictions=1000,
    ) -> List[int]:  # is this fhe, fl or both ?
        """_summary_

        Args:
            dataset (str): _description_
            nb_classes (int): _description_
            nb_teachers (int): _description_
            job_uuid (str): _description_
            number_of_predictions (int, optional): _description_. Defaults to 1000.
        """
        # FIXME: uncomment when using separate hosts
        # receive_keys()

        self.nb_classes: int = nb_classes
        self.nb_teachers: int = nb_teachers

        logger.info("Retrieving public dataset: %s", dataset)
        data, truth_one_hot = self.usecase.get_dataset(dataset, number_of_predictions)
        truth = [list(label).index(1) for label in truth_one_hot]

        logger.info("Looking for model in %s", self.models_path)
        model_loaded, model_fname = self.book_model_directory(self.models_path)
        if not model_loaded:
            raise FileNotFoundError("No free model found in ", self.models_path)
        logger.info("Loading model: %s", model_fname)
        model = self.usecase.get_model(model_fname)

        logger.info("Labeling the dataset")
        predictions: List[int] = self.usecase.do_prediction(model, data)
        logger.debug("Dataset : %s", data)
        logger.debug(predictions)
        logger.debug("Truth : %s", truth)

        self.remove_lock(model_fname)

        return predictions

    def send_predictions(
        self,
        predictions: List[int],
        job_uuid: str,
        encrypt: bool,
        differential_privacy_noise_parameter=0,
    ) -> None:
        """_summary_

        Args:
            predictions (List[int]): _description_
            job_uuid (str): _description_
        """
        topic_name = "votes_pate"
        if encrypt:
            topic_name = "vote_he"

        if encrypt:
            logger.info("Encrypting the labels")
            # This is the starting point of the encryptions
            encryptions_start_time = time.time()

            teacher_data_directory: Path = Path("/app_saferlearn/pate_he/teacher/data")
            fhe_encryption_executable_folder: Path = Path(
                "/app_saferlearn/pate_he/teacher/bin"
            )
            prediction_bytes: List[bytes] = self.encrypt_votes(
                predictions,
                self.nb_classes,
                RPYC_HOST,
                teacher_data_directory,
                fhe_encryption_executable_folder,
                differential_privacy_noise_parameter,
            )

            # The predictions are over
            encryptions_end_time = time.time()

            encryptions_computation_time = encryptions_end_time - encryptions_start_time
            logger.info("Teacher encryptions took %s s", encryptions_computation_time)
        else:
            prediction_bytes = [str(x).encode("utf-8") for x in predictions]

        logger.info("Sending the labels")
        self._producer = utilities.connect_kafka_producer()
        for sample, he_vote_bytes in enumerate(prediction_bytes):
            utilities.publish_message(
                self._producer,
                topic_name,
                '{"key": "' + str(sample) + '", "worker": "' + str(RPYC_HOST) + '"}',
                he_vote_bytes,
                encode=False,
            )

        logger.info("Calling finished_job")
        if ORCHESTRATOR_HOST is not None and RPYC_PORT is not None:
            utilities.register_finish(
                ORCHESTRATOR_HOST,
                ORCHESTRATOR_PORT,
                self.worker_uuid,
                job_uuid,
                RPYC_HOST,
                RPYC_PORT,
            )
        if self._producer is not None:
            self._producer.close()
        return

    def encrypt_votes(
        self,
        selected_class_indices: List[int],
        number_of_classes: int,
        teacher_id: str,
        teacher_data_directory: Path,
        fhe_encryption_executable_folder: Path,
        differential_privacy_noise_parameter=0.0,
    ) -> List[bytes]:
        """_summary_

        Args:
            selected_class_indices (List[int]): _description_
            number_of_classes (int): _description_
            teacher_id (str):  Teacher's identification (IP)
            teacher_data_directory (Path): _description_
            fhe_encryption_executable_folder (Path): _description_

        Returns:
            List[bytes]: _description_
        """
        logger.info("Encrypting votes")
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
            if differential_privacy_noise_parameter != 0:
                logger.info("Use noise-friendly HE encryption")
                logger.debug(
                    "Differential privacy parameter : %s",
                    differential_privacy_noise_parameter,
                )
                fhe_encryption_executable = fhe_encryption_executable_folder / "encrypt"
            else:
                logger.info("Use classic HE encryption")
                fhe_encryption_executable = (
                    fhe_encryption_executable_folder / "encrypt_no_dp"
                )
            executable_parameters = [
                fhe_encryption_executable,
                teacher_data_directory / "keys/pub_params",
                teacher_data_directory / "keys/pub_key",
                str(self.nb_teachers),
                str(self.nb_classes),
                plain_vote_folder / str(vote_position),
                encrypted_vote_folder / str(vote_position),
            ]
            if differential_privacy_noise_parameter != 0:
                # FIXME: do we just use a "safe" amplitude ? which one ?
                # or do we clamp the noise produced to be super-safe ?
                differential_privacy_noise_amplitude = math.ceil(
                    5 * differential_privacy_noise_parameter
                )
                executable_parameters = (
                    executable_parameters[:5]
                    + [str(differential_privacy_noise_amplitude)]
                    + executable_parameters[5:]
                )

            try:
                encryption_subprocess: subprocess.CompletedProcess = subprocess.run(
                    executable_parameters,
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

    def write_mpc_votes(self, votes: List[int], dataset: str, client_id: int) -> None:
        """_summary_

        Args:
            votes (List[int]): _description_
            dataset (str): _description_
            client_id (int): _description_
        """
        logger.debug("Writing votes in file for MPC")
        dataset_dir: Path = Path("/opt/input-data/" + dataset)
        try:
            dataset_dir.mkdir(parents=True, exist_ok=True)
        except OSError as error:
            logger.error("Error while creating directory: %s", error)
        with open(
            dataset_dir / str(client_id), "w", encoding="utf-8"
        ) as plaintext_file:
            for vote in votes:
                plaintext_file.write(str(vote) + "\n")


def launch_rpyc_server(
    hostname: str,
    port: str,
    worker_uuid: str,
    auto_register: bool,
    protocol_config: Dict[str, Any],
    backlog: int,
    number_of_computing_parties: int,
    size_of_batches: int,
    program: str,
    tag: str = "",
    nb_teachers: int = 3,
    sigma: int = 1,
    rounds: int = -1,
) -> None:
    """_summary_

    Args:
        hostname (str): _description_
        port (str): _description_
        worker_uuid (str): _description_
        auto_register (bool): _description_
        protocol_config (Dict[str, Any]): _description_
        backlog (int): _description_
        number_of_computing_parties (int): _description_
        size_of_batches (int): _description_
        program (str): _description_
        tag (str, optional): _description_. Defaults to "".
        nb_teachers (int, optional): _description_. Defaults to 3.
        sigma (int, optional): _description_. Defaults to 1.
        rounds (int, optional): _description_. Defaults to -1.
    """
    server.ThreadedServer(
        WorkerService(
            worker_uuid,
            number_of_computing_parties,
            size_of_batches,
            program,
            tag,
            nb_teachers,
            sigma,
            rounds,
        ),
        hostname=hostname,
        port=int(port),
        auto_register=auto_register,
        protocol_config=protocol_config,
        backlog=backlog,
    ).start()


def main() -> None:
    """TODO: Docstring for main.
    :returns: TODO

    """
    parser: argparse.ArgumentParser = argparse.ArgumentParser()
    parser.add_argument(
        "-t", "--tag", default="Radio", help="Node type, default 'Radio'"
    )
    parser.add_argument(
        "-N",
        "--computing-parties-number",
        default="3",
        help="Number of computing parties, default 3",
    )
    parser.add_argument(
        "-B", "--size-batch", default="10", help="Size of batch, default 10"
    )
    parser.add_argument(
        "program", default="pate-teacher", help="Program to run on the teachers"
    )

    parser.add_argument("--nteachers", help="Number of teachers")
    parser.add_argument("--sigma", help="Sigma for Gaussian noise distribution")
    parser.add_argument("--rounds", help="Number of rounds for noise")

    args: argparse.Namespace = parser.parse_args()

    tag: str = args.tag
    number_of_computing_parties: int = args.computing_parties_number
    size_of_batches: int = args.size_batch
    program_teacher: str = args.program

    protocol_config: Dict[str, bool] = dict(
        instantiate_custom_exceptions=True, import_custom_exceptions=True
    )

    logger.info("Generating UUID")
    worker_uid: str = utilities.generate_uuid()
    logger.debug("Launching Worker RPyC")
    threads: List[threading.Thread] = []
    if tag == "privacy_guardian":
        logger.debug("Launching privacy guardian")
        teacher_subprocess: threading.Thread = threading.Thread(
            target=launch_rpyc_server,
            args=(
                RPYC_HOST,
                RPYC_PORT,
                worker_uid,
                False,
                protocol_config,
                500,
                number_of_computing_parties,
                size_of_batches,
                program_teacher,
                tag,
                args.nteachers,
                args.sigma,
                args.rounds,
            ),
        )
    else:
        logger.debug("Launching data owner")
        teacher_subprocess = threading.Thread(
            target=launch_rpyc_server,
            args=(
                RPYC_HOST,
                RPYC_PORT,
                worker_uid,
                False,
                protocol_config,
                500,
                number_of_computing_parties,
                size_of_batches,
                program_teacher,
            ),
        )
    logger.debug("Starting data owner with %s datatype", tag)
    heartbeat_subprocess: threading.Thread = threading.Thread(
        target=utilities.heartbeat,
        args=(
            RPYC_HOST,
            RPYC_PORT,
            worker_uid,
            tag,
            5,
        ),
    )
    threads.append(teacher_subprocess)
    threads.append(heartbeat_subprocess)
    teacher_subprocess.start()

    logger.debug("Registering worker")
    heartbeat_subprocess.start()
    # launch a ssh server (for federated learning)
    ssh_subprocess: subprocess.CompletedProcess = subprocess.run(
        ["/usr/sbin/sshd", "-D"],
        capture_output=True,
        check=True,
    )
    utilities.log_subprocess(ssh_subprocess)


if __name__ == "__main__":
    main()
