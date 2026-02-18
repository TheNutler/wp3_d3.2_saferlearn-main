#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File              : orchestrator.py
# Author            : hargathor <3949704+hargathor@users.noreply.github.com>
# Date              : 14.10.2022
# Last Modified Date: 14.10.2022
# Last Modified By  : hargathor <3949704+hargathor@users.noreply.github.com>
"""Orchestrator module"""

import concurrent.futures
import json
import logging
import math
import os
import pathlib
import shutil
import subprocess
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import torchvision

import numpy as np
import requests
import rpyc
from kafka import KafkaConsumer, KafkaProducer
from kafka.errors import KafkaError
from rpyc.core import GenericException

import utilities
from saferlearn import Dataset

logging.basicConfig(level=logging.DEBUG)
logger: logging.Logger = logging.getLogger("orchestrator")
logger.setLevel(logging.DEBUG)

ORCHESTRATOR_HOST: Optional[str] = os.environ.get("ORCHESTRATOR_HOST", "127.0.0.1")
ORCHESTRATOR_PORT: Optional[str] = os.environ.get("ORCHESTRATOR_PORT", "5000")


class Job:
    """_summary_"""

    def __init__(
        self,
        workers,
        nb_classes,
        algorithm,
        data_type="Radio",
        dataset="public-dataset",
    ) -> None:
        self.uuid: str = str(uuid.uuid4())
        self.params: Dict[str, Any] = {
            "algorithm": algorithm,
            "nb_classes": nb_classes,
            "nb_teachers": len(workers),
            "dataset": dataset,
        }
        self.data_type: str = data_type
        self.workers = workers
        logger.debug("Job creation finished for job %s", self.__dict__)

    def __str__(self) -> str:
        return json.dumps(vars(self), ensure_ascii=False)

    def __repr__(self) -> str:
        return self.__str__()

    def to_json(self) -> str:
        """JSON representation

        Returns:
            str: The stringified JSON representation of the Job
        """
        return str(self)

    # integrate FL
    def create_job_fl(self, tff_central: Dict[str, Any]) -> None:
        """Launch a MPC job

        Args:
            tff_central Dict[str, Any]: The fl aggregator
            dataset (str, optional): The dataset used by data owners.
            Defaults to "cifar10_train".
        """
        logger.error("FL is not Implemented")
        update_job_state(self.uuid, "done")

    def create_job(self, nb_samples, differential_privacy_parameter: float) -> None:
        """Launch a FHE job

        Args:
            dataset (str): _description_
        """
        self.send_parameters()
        nb_labeled_samples = -1

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=len(self.workers)
        ) as executor:
            future_to_worker = {
                executor.submit(
                    self.rpyc_call,
                    worker,
                    self.params["nb_classes"],
                    self.params["nb_teachers"],
                    self.params["dataset"],
                    True,
                    nb_samples,
                    differential_privacy_parameter,
                ): worker
                for worker in self.workers
            }
            for future in concurrent.futures.as_completed(future_to_worker):
                worker = future_to_worker[future]
                try:
                    nb_labeled_samples_by_worker = future.result()
                    logger.info(
                        "%s has labeled %d samples",
                        worker,
                        nb_labeled_samples_by_worker,
                    )
                    if nb_labeled_samples == -1:
                        nb_labeled_samples = nb_labeled_samples_by_worker
                    elif nb_labeled_samples != nb_labeled_samples_by_worker:
                        logger.error(
                            "%r has labeled %d samples while another has labeled %d",
                            worker,
                            nb_labeled_samples_by_worker,
                            nb_labeled_samples,
                        )
                except Exception as exc:
                    logger.error("%r generated an exception: %s", worker, exc)
        logger.info("Teachers have labeled the public dataset")
        start: float = time.time()
        encrypted_aggregation_result: List[Tuple[int, bytes]]
        encrypted_aggregation_result = self.aggregation_tfhe(
            len(self.workers),
            self.params["nb_classes"],
            nb_labeled_samples,
            differential_privacy_parameter,
        )
        end: float = time.time()
        logger.info("Aggregation took %ss", end - start)
        logger.info("Cleaning Topic")
        utilities.clean_topic("model_he")
        kafka_producer: KafkaProducer = utilities.connect_kafka_producer()
        logger.info("Publishing results")
        for vote_id, encrypted_vote in encrypted_aggregation_result:
            utilities.publish_message(
                kafka_producer, "model_he", vote_id, encrypted_vote, encode=False
            )
        utilities.publish_message(
            kafka_producer, "nb_votes", "number_votes", nb_labeled_samples
        )
        update_job_state(self.uuid, "done")

    def create_job_clear(
        self,
        nb_samples: int,
        differential_privacy_parameter: float,
        api_host: str = "127.0.0.1",
        api_port: int = 5000,
    ) -> None:
        """Launch a PATE job

        Args:
            dataset (str): _description_
        """
        self._api_host = api_host
        self._api_port = api_port
        self.send_parameters()
        # threads: List[threading.Thread] = []
        # active_threads: int = threading.active_count()

        nb_labeled_samples = -1

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=len(self.workers)
        ) as executor:
            future_to_worker = {
                executor.submit(
                    self.rpyc_call,
                    worker,
                    self.params["nb_classes"],
                    self.params["nb_teachers"],
                    self.params["dataset"],
                    False,
                    nb_samples,
                    differential_privacy_parameter,
                ): worker
                for worker in self.workers
            }
            for future in concurrent.futures.as_completed(future_to_worker):
                worker = future_to_worker[future]
                try:
                    nb_labeled_samples_by_worker = future.result()
                    logger.info(f"{worker} has labeled {nb_labeled_samples} samples")
                    if nb_labeled_samples == -1:
                        nb_labeled_samples = nb_labeled_samples_by_worker
                    elif nb_labeled_samples != nb_labeled_samples_by_worker:
                        logger.error(
                            f"{worker} has labeled {nb_labeled_samples_by_worker} samples while another has labeled %{nb_labeled_samples}",
                        )
                    else:
                        nb_labeled_samples += nb_labeled_samples
                except Exception as exc:
                    logger.error(f"{worker} generated an exception: {exc}")

        logger.info("Teachers have labeled the public dataset")
        logger.debug(f"Total number of labeled samples is {nb_labeled_samples}")
        start: float = time.time()
        aggregation_result = self.aggregation(
            int(self.params["nb_teachers"]),
            int(self.params["nb_classes"]),
            nb_labeled_samples_by_worker,  # TODO: find a proper way to know the real count of samples without downloading the dataset
            differential_privacy_parameter,
        )
        end: float = time.time()
        topic_name = "model_pate"
        logger.info("Aggregation took %ss", end - start)
        logger.info("Cleaning Topic")
        utilities.clean_topic(topic_name)
        kafka_producer: KafkaProducer = utilities.connect_kafka_producer()
        logger.info("Publishing result")
        # logger.info(f"Aggregation result: {aggregation_result}")
        from alive_progress import alive_bar

        with alive_bar(len(aggregation_result), bar="bubbles") as bar:
            for vote_id, vote in enumerate(aggregation_result):
                utilities.publish_message(
                    kafka_producer, topic_name, vote_id, vote, encode=True
                )
                bar()
            utilities.publish_message(
                kafka_producer, "nb_votes", "number_votes", nb_labeled_samples
            )
        update_job_state(self.uuid, "done")

    def create_job_mpc(self, computing_parties: List[Dict[str, Any]]) -> None:
        """Launch a MPC job

        Args:
            workers (List[Dict[str, Any]]): The list of workers
            computing_parties (List[Dict[str, Any]]): The list of computing parties
            workers
            dataset (str, optional): The dataset used by data owners. Defaults to
            "StubDataset".
        """
        params: Dict[str, Any] = {}
        params["algorithm"] = "mpc"
        self.send_parameters()
        threads: List[threading.Thread] = []
        hosts: List[str] = []
        host_file: Path = Path("./data/app_saferlearn/HOSTS")
        host_file.touch(exist_ok=True)
        with open(host_file, "r", encoding="utf-8") as file_hosts:
            for line in file_hosts:
                hosts.append(line.split(":")[0])
        threads_computing_parties: List[threading.Thread] = []
        logger.debug("Launching computing parties: %s", computing_parties)
        for computing_party in computing_parties:
            mpc_aggregator_thread: threading.Thread = threading.Thread(
                target=self.rpyc_call_computing_party,
                args=(computing_party, computing_parties, len(self.workers)),
            )
            threads_computing_parties.append(mpc_aggregator_thread)
            mpc_aggregator_thread.start()
        # time.sleep(5)
        first: bool = True
        client_id: int = 0
        last_flag: int = 0
        last_id: int = len(self.workers) - 1
        logger.debug("Launching %s teachers", len(self.workers))
        logger.debug("Computing parties: %s", computing_parties)
        for worker in self.workers:
            # here add the client id & the last flag
            if client_id == last_id:
                last_flag = 1
            mpc_aggregator_thread = threading.Thread(
                target=self.rpyc_call_mpc,
                args=(
                    worker,
                    self.params["dataset"],
                    computing_parties,
                    first,
                    str(client_id),
                    str(last_flag),
                ),
            )
            first = False
            client_id += 1
            threads.append(mpc_aggregator_thread)
            mpc_aggregator_thread.start()
        while len(self.get_workers()) > 0:
            time.sleep(5)
            logger.debug("Waiting for workers....")
            logger.debug("Remaining workers %s", len(self.get_workers()))
        logger.info("#############################  Workers are done")
        update_job_state(self.uuid, "done")

    def receive_keys(self) -> None:
        """Receive HE keys"""
        topic_name: str = "keys"
        keys_dir: Path = Path("/app_saferlearn/pate_he/aggregator/data/keys")
        kafka_consumer: KafkaConsumer = utilities.connect_kafka_consumer(
            topic_name, group_id="orchestrator"
        )

        utilities.receive_file_kafka(kafka_consumer, keys_dir, "tgsw_params_1")
        utilities.receive_file_kafka(kafka_consumer, keys_dir, "tgsw_params_2")
        utilities.receive_file_kafka(kafka_consumer, keys_dir, "boot_key_1")
        utilities.receive_file_kafka(kafka_consumer, keys_dir, "boot_key_2")

        kafka_consumer.close()

    def send_parameters(self) -> None:
        """Send computation parameters to each agent"""
        topic_name: str = "params"
        kafka_producer: KafkaProducer = utilities.connect_kafka_producer()
        logger.debug("Sending params: ")
        logger.debug(self.params)
        for key, value in self.params.items():
            utilities.publish_message(kafka_producer, topic_name, str(key), str(value))
        if self.params["algorithm"] == "fhe":
            self.receive_keys()
        utilities.clean_topic(topic_name)

    # integrate FL
    def rpyc_call_fl(
        self,
        tff_central: Dict[str, Any],
        workers: List[Dict[str, Any]],
        dataset: str,
    ) -> None:
        """_summary_

        Args:
            tff_central (Dict[str, Any]): _description_
            workers (List[Dict[str, Any]]): _description_
        """
        logger.debug(tff_central)
        conn: rpyc.Connection = rpyc.connect(
            tff_central["ip"],
            tff_central["port"],
            config={"sync_request_timeout": None},
        )
        try:
            logging.error("Not implemented")
            # TODO: implement a call to an exposed FL method/framework
            # conn.root.exposed_launch_tff_central(workers, dataset)

        except GenericException as err:
            logger.warning("Timeout as expected %s", err)

    def rpyc_call_mpc(
        self,
        worker: Dict[str, Any],
        dataset: str,
        computing_parties: List[Dict[str, str]],
        first: bool,
        client_id: int,
        last_flag: int,
    ) -> None:
        """_summary_

        Args:
            worker (Dict[str, Any]): _description_
            dataset (str): _description_
            computing_parties (List[Dict[str, Any]]): _description_
            first (bool): _description_
            client_id (int): _description_
            last_flag (int): _description_
        """
        conn: rpyc.Connection = rpyc.connect(
            worker["ip"], worker["port"], config={"sync_request_timeout": None}
        )
        default_dataset: str = "public-dataset"
        try:
            conn.root.exposed_get_predictions_mpc(
                dataset,
                default_dataset,
                json.dumps(computing_parties),
                client_id,
                self.uuid,
                last_flag,
            )
            if first:
                update_job_state(self.uuid, conn.root.exposed_get_state())
        except GenericException as err:
            logger.warning("Timeout as expected %s", err)

    def rpyc_call_computing_party(
        self,
        computing_party: Dict[str, Any],
        computing_parties: List[Dict[str, Any]],
        nb_workers: int,
    ) -> None:
        """_summary_

        Args:
            computing_party (Dict[str, Any]): _description_
            computing_parties (List[Dict[str, Any]]): _description_
            nb_workers (int): _description_
        """
        conn: rpyc.Connection = rpyc.connect(
            computing_party["ip"],
            computing_party["port"],
            config={"sync_request_timeout": None},
        )
        try:
            conn.root.exposed_launch_computing_party(computing_parties, nb_workers)
        except GenericException as err:
            logger.warning("Timeout as expected %s", err)

    def rpyc_call(
        self,
        worker: Dict[str, Any],
        nb_classes: int,
        nb_teachers: int,
        dataset: str,
        encrypt: bool,
        nb_samples: int,
        differential_privacy_noise_parameter: float,
    ) -> int:
        """_summary_

        Args:
            worker (Dict[str, Any]): _description_
            nb_classes (int): _description_
            nb_teachers (int): _description_
            dataset (str): Dataset name
        """
        logger.debug(worker)
        conn: rpyc.Connection = rpyc.connect(
            worker["ip"], worker["port"], config={"sync_request_timeout": None}
        )
        try:
            nb_samples_labeled = conn.root.exposed_label_dataset(
                dataset,
                nb_classes,
                nb_teachers,
                self.uuid,
                encrypt,
                nb_samples,
                differential_privacy_noise_parameter,
            )
            logger.debug(f"{nb_samples_labeled} samples has been labeled")
            return nb_samples_labeled
        except GenericException:
            logger.warning("Timeout as expected")
        # return 0

    def get_workers(self) -> List[str]:
        """_summary_

        Returns:
            _type_: _description_
        """
        return utilities.get_list_of_workers(self.uuid)

    def generate_dp_noise_file(self, nb_samples, nb_classes, sigma):
        """Generate the correct file format for noise in pate_he_v2.3

        Args:
            nb_samples (_type_): _description_
            nb_classes (_type_): _description_
            sigma (_type_): _description_
        """
        logger.info("Generating clear noise for DP HE aggregation")
        noise_file_path = pathlib.Path("/tmp/noise")
        with open(noise_file_path, "w", encoding="utf-8") as noise_file:
            for _ in range(2):
                # This is because aggregate_several in pate_he_v2.3 is weird
                # these noise values are actually not used
                noise = np.random.normal(0, sigma, nb_classes)
                noise_file.write(
                    " ".join(["{:f}".format(abs(x)) for x in noise]) + "\n"
                )
            logger.debug("Noise generated :")
            for _ in range(nb_samples):
                noise = np.random.normal(0, sigma, nb_classes)
                line = " ".join(["{:f}".format(abs(x)) for x in noise])
                logger.debug(line)
                noise_file.write(line + "\n")
        return noise_file_path

    def aggregation_tfhe(
        self,
        nb_teachers: int,
        nb_classes: int,
        nb_samples: int,
        differential_privacy_noise_parameter=0.0,
    ) -> List[Tuple[int, bytes]]:
        """_summary_

        Args:
            nb_teachers (int): _description_
            nb_classes (int): _description_

        Returns:
            Tuple[List[bytes], int]: _description_
        """
        topic_name: str = "vote_he"
        aggregator_dir: Path = Path("/app_saferlearn/pate_he/aggregator")
        data_dir: Path = aggregator_dir / "data"
        ciphertext_dir: Path = data_dir / "ciphertexts"
        votes_dir: Path = ciphertext_dir / "encrypted_teacher_votes"
        result_dir: Path = ciphertext_dir / "encrypted_argmax_folder"
        parameters_dir: Path = data_dir / "keys"
        encrypted_argmax_dataset: List[Tuple[int, bytes]] = []
        for sub_dir in votes_dir.iterdir():
            logger.debug("Cleaning up %s", sub_dir)
            try:
                shutil.rmtree(sub_dir)
            except NotADirectoryError:
                logger.error("Not a directory, skipping")

        total_number_of_votes_expected = nb_samples * nb_teachers

        votes: List[Dict[str, Any]] = []
        try:
            consumer: KafkaConsumer = utilities.connect_kafka_consumer(
                topic_name,
                "orchestrator",
                consumer_timeout_ms=1000,
                enable_auto_commit=True,
            )
            received_votes = 0
            logger.debug(
                "Expecting to receive %d votes", total_number_of_votes_expected
            )
            while received_votes < total_number_of_votes_expected:
                records = consumer.poll(timeout_ms=1000)
                if not records:
                    time.sleep(2)
                else:
                    for _, consumer_records in records.items():
                        for msg in consumer_records:
                            vote = {}
                            key: Dict[str, Any] = json.loads((msg.key).decode("utf-8"))
                            logger.debug("Received key %s", key)
                            vote: Dict[str, Any] = {
                                "data": msg.value,
                                "key": int(key["key"]),
                                "worker": key["worker"],
                            }
                            votes.append(vote)
                            received_votes += 1
            consumer.close()
            logger.debug("Received %d votes", received_votes)
        except KafkaError as err:
            logger.error("Kafka error: %s", err)

        logger.info("Number of votes %s", len(votes))
        for vote in votes:
            vote_dir: Path = votes_dir / str(vote.get("key"))
            try:
                vote_dir.mkdir(parents=True, exist_ok=True)
            except OSError as error:
                logger.error("Error while creating directory: %s", error)
            current_vote_path: Path = vote_dir / str(vote.get("worker"))
            logger.debug("Writing vote %s", current_vote_path)
            with open(current_vote_path, "wb") as encrypted_votes_file:
                ciphertext: Any = vote.get("data")
                if ciphertext is not None:
                    encrypted_votes_file.write(ciphertext)
            encrypted_votes_file.close()

        # Batching
        votes_paths = votes_dir.iterdir()
        batch_size = 100
        for index, vote_path in enumerate(
            sorted(votes_paths, key=lambda path: int(path.stem))
        ):
            batch_index = index // batch_size
            batch_path = votes_dir / f"batch_{batch_index}"
            batch_path.mkdir(parents=True, exist_ok=True)
            # vote_path.rename(batch_path / vote_path.stem)
            vote_path.rename(batch_path / str(int(vote_path.stem) % batch_size))

        fhe_aggregation_executable_folder = aggregator_dir / "bin"
        if differential_privacy_noise_parameter != 0:
            logger.info("Use DP HE aggregation")
            logger.debug(
                "Differential privacy parameter : %s",
                differential_privacy_noise_parameter,
            )
            fhe_encryption_executable = (
                fhe_aggregation_executable_folder / "aggregate_several"
            )
        else:
            logger.info("Use classic HE aggregation")
            fhe_encryption_executable = (
                fhe_aggregation_executable_folder / "aggregate_several_no_dp"
            )

        executable_parameters = [
            fhe_encryption_executable,
            parameters_dir / "tgsw_params_1",
            parameters_dir / "tgsw_params_2",
            str(nb_teachers),
            str(nb_classes),
            parameters_dir / "boot_key_1",
            parameters_dir / "boot_key_2",
            result_dir,
            votes_dir,
        ]
        if differential_privacy_noise_parameter != 0:
            # FIXME: do we just use a "safe" amplitude ? which one ?
            # or do we clamp the noise produced to be super-safe ?
            differential_privacy_noise_amplitude = math.ceil(
                5 * differential_privacy_noise_parameter
            )
            noise_file = self.generate_dp_noise_file(
                batch_size,
                nb_classes,
                differential_privacy_noise_parameter,
            )
            executable_parameters.append(str(noise_file))
            executable_parameters.append(str(differential_privacy_noise_amplitude))
        try:
            # Start the he aggregation
            he_aggregation_start_time = time.time()
            for index, batch in enumerate(
                sorted(votes_dir.iterdir(), key=lambda path: int((path.stem)[6:]))
            ):
                logger.debug("Aggregating %s of size %s", batch.stem, batch_size)

                if differential_privacy_noise_parameter != 0 and index != 0:
                    noise_file = self.generate_dp_noise_file(
                        batch_size,
                        nb_classes,
                        differential_privacy_noise_parameter,
                    )
                batch_result_dir = result_dir / batch.stem
                batch_result_dir.mkdir(exist_ok=True)
                executable_parameters[7] = str(batch_result_dir)
                executable_parameters[8] = str(batch)

                aggregate_subprocess: subprocess.CompletedProcess = subprocess.run(
                    executable_parameters,
                    capture_output=True,
                    check=True,
                )
                utilities.log_subprocess(aggregate_subprocess)
            # The aggregation is over
            he_aggregation_end_time = time.time()

            he_aggregation_time = he_aggregation_end_time - he_aggregation_start_time
            logger.info("HE aggregation took %s s", he_aggregation_time)
        except subprocess.CalledProcessError as process_exception:
            utilities.log_subprocess(process_exception)

        for batch in result_dir.iterdir():
            for sub_dir in batch.iterdir():
                sample_index = int(sub_dir.stem) + batch_size * int(batch.stem[6:])
                logger.debug(
                    "Preparing to send encrypted argmax n°%d",
                    sample_index,
                )
                with open(sub_dir, "rb") as encrypted_argmax_file:
                    encrypted_argmax_dataset.append(
                        (sample_index, encrypted_argmax_file.read())
                    )

        return encrypted_argmax_dataset

    def pate_aggregate(self, index: int, inputs, nb_classes: int, sigma=0.0):
        """PATE aggregation

        Args:
            index (int): _description_
            inputs (_type_): _description_
            nb_classes (int): _description_
            sigma (float): _description_

        Returns:
            _type_: _description_
        """
        # logger.info("Clear PATE aggregation for sample n°%d", index)
        # logger.debug("nb_classes : %s", nb_classes)
        # logger.debug("Votes : %s", inputs)
        score = [0] * nb_classes
        for vote in inputs:
            score[vote] += 1
        # logger.debug("Clear votes counting : %s", score)
        if sigma != 0:
            logger.info("Use differential privacy")
            logger.debug("Differential privacy parameter : %s", sigma)
            noise = np.random.normal(0, sigma, nb_classes)
            logger.debug("Clear noise vector : %s", noise)
            score = list(score + noise)
            logger.debug("Clear noisy votes counting : %s", score)
        result = score.index(max(score))
        # logger.debug("Winning class : %s", result)
        return result

    def aggregation(
        self,
        nb_teachers: int,
        nb_classes: int,
        nb_samples: int,
        differential_privacy_parameter=0.0,
    ) -> List[Tuple[int, bytes]]:
        """_summary_

        Args:
            nb_teachers (int): _description_
            nb_classes (int): _description_

        Returns:
            Tuple[List[bytes], int]: _description_
        """
        # topic_name: str = "votes_pate"
        topic_name: str = "votes_pate"
        # FIXME: is the stuff below needed ?
        # aggregator_dir: Path = Path("/tmp/app_saferlearn/pate_he/aggregator")
        # data_dir: Path = aggregator_dir / "data"
        # ciphertext_dir: Path = data_dir / "ciphertexts"
        # votes_dir: Path = ciphertext_dir / "encrypted_teacher_votes"
        # try:
        #     for sub_dir in votes_dir.iterdir():
        #         logger.debug("Cleaning up %s", sub_dir)
        #         try:
        #             shutil.rmtree(sub_dir)
        #         except NotADirectoryError:
        #             logger.error("Not a directory, skipping")
        # except FileNotFoundError:
        #     logger.error(f"File not found {sub_dir}")

        logger.debug(f"nb_samples: {nb_samples}")
        logger.debug(f"nb_teachers: {nb_teachers}")
        total_number_of_votes_expected = nb_samples * nb_teachers
        logger.debug(f"Expecting {total_number_of_votes_expected} votes")

        votes: List[Dict[str, Any]] = []
        try:
            consumer: KafkaConsumer = utilities.connect_kafka_consumer(
                topic_name,
                "orchestrator",
                consumer_timeout_ms=1000,
                enable_auto_commit=False,
            )
            consumer.subscribe(topic_name)
            received_votes = 0
            from alive_progress import alive_bar

            running = True
            with alive_bar(total_number_of_votes_expected, bar="bubbles") as bar:
                while running:
                    records = consumer.poll(timeout_ms=1000)
                    if records is None:
                        continue
                    elif received_votes >= total_number_of_votes_expected:
                        break
                    else:
                        for _, consumer_records in records.items():
                            for msg in consumer_records:
                                vote = {}
                                key: Dict[str, Any] = json.loads(
                                    (msg.key).decode("utf-8")
                                )
                                # logger.debug("Received key %s", key)
                                vote: Dict[str, Any] = {
                                    "data": int(msg.value),
                                    "key": int(key["key"]),
                                    "worker": key["worker"],
                                }
                                votes.append(vote)
                                received_votes += 1
                                bar()
        except KafkaError as err:
            logger.error("Kafka error: %s", err)
        finally:
            # Close down consumer to commit final offsets.
            consumer.close()
        utilities.clean_topic(topic_name)
        logger.debug(f"Received votes from kafka: {received_votes}")
        # Start the clear aggregation
        clear_aggregation_start_time = time.time()

        sorted_votes = [
            [0 for teacher in range(nb_teachers)] for sample in range(nb_samples)
        ]
        sorted_worker = []
        try:
            for vote in votes:
                if vote["worker"] not in sorted_worker:
                    sorted_worker.append(vote["worker"])
                sorted_votes[vote["key"]][sorted_worker.index(vote["worker"])] = vote[
                    "data"
                ]
        except IndexError as err:
            logger.error(err)

        # logger.debug("%s", sorted_votes)
        aggregated_votes = []
        from alive_progress import alive_bar

        with alive_bar(nb_samples, bar="bubbles") as bar:
            for i in range(nb_samples):
                winning_class = self.pate_aggregate(
                    i, sorted_votes[i], nb_classes, differential_privacy_parameter
                )
                aggregated_votes.append(winning_class)
                bar()
        logger.debug(f"nb_teachers : {nb_teachers}, nb samples: {nb_samples}")
        # logger.debug(f"Aggregated votes: {aggregated_votes}")

        # The computation is over
        clear_aggregation_end_time = time.time()

        clear_aggregation_time = (
            clear_aggregation_end_time - clear_aggregation_start_time
        )
        logger.info("Clear aggregation took %s s", clear_aggregation_time)
        return aggregated_votes


def update_job_state(job_uuid: str, state: str) -> Dict[str, Any]:
    """_summary_

    Args:
        job_uuid (str): _description_
        state (str): _description_

    Returns:
        _type_: _description_
    """
    url: str = f"http://{ORCHESTRATOR_HOST}:{5000}/job/{job_uuid}/{state}"
    logger.debug(url)

    try:
        resp: requests.Response = requests.put(url, timeout=10)
        if resp.status_code != 200:
            logger.error("Error in HTTP request: %s", resp.json())
            return resp.json()
    except ConnectionError as err:
        logger.error("Connection error: %s", err)
    return {}
