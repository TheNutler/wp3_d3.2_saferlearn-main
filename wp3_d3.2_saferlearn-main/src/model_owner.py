#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File              : model_owner.py
# Author            : hargathor <3949704+hargathor@users.noreply.github.com>
# Date              : 12.04.2019
# Last Modified Date: 29.04.2022
# Last Modified By  : hargathor <3949704+hargathor@users.noreply.github.com>

"""Model owner module"""

import logging
import subprocess
import time
from pathlib import Path
from typing import Dict, List

from kafka import KafkaConsumer, KafkaProducer

import utilities

logging.basicConfig(level=logging.DEBUG)
logger: logging.Logger = logging.getLogger("model_owner")
logger.setLevel(logging.DEBUG)
logging.getLogger("kafka").setLevel(logging.WARNING)


def retrieve_params() -> Dict[str, str]:
    """Retrieve the run parameters from the orchestrator

    Returns:
        Dict[str, str]: The dictionary of parameters
    """
    logger.info("Waiting for parameters...")
    topic_name: str = "params"
    params: Dict[str, str] = {}
    group_id: str = "student"
    consumer: KafkaConsumer = utilities.connect_kafka_consumer(topic_name, group_id)
    logger.debug("polling")
    for msg in consumer:
        key: str = msg.key.decode("utf-8")
        value: str = msg.value.decode("utf-8")
        logger.debug("Parameter %s : %s ", value, key)
        params[key] = value
        if "algorithm" in params:
            if params["algorithm"] == "mpc" or params["algorithm"] == "pate":
                break
            elif params["algorithm"] == "fhe":
                if "nb_teachers" in params and "nb_classes" in params:
                    break
            else:
                logger.debug("FL not supported")
                break
    consumer.close()
    utilities.clean_topic(topic_name)

    return params


def share_he_keys() -> None:
    """Share HE keys and parameters to partners"""
    topic_name: str = "keys"
    fhe_dir: Path = Path("/app_saferlearn/pate_he/")
    student_dir: Path = fhe_dir / "student/"
    keys_dir: Path = student_dir / "data/keys/"
    teacher_keys_dir: Path = fhe_dir / "teacher/data/keys/"

    kafka_producer: KafkaProducer = utilities.connect_kafka_producer()

    # TODO : share the keys through Kafka when no longer sharing src/ in volume
    # utilities.send_file_kafka(kafka_producer, keys_dir, "pub_key", topic_name)
    # utilities.send_file_kafka(kafka_producer, keys_dir, "pub_params", topic_name)
    try:
        output: subprocess.CompletedProcess = subprocess.run(
            ["cp", keys_dir / "pub_key", keys_dir / "pub_params", teacher_keys_dir],
            capture_output=True,
            check=True,
        )
        utilities.log_subprocess(output)
        logger.info("Shared keys to the teachers")
    except subprocess.CalledProcessError as process_exception:
        utilities.log_subprocess(process_exception)

    utilities.send_file_kafka(kafka_producer, keys_dir, "tgsw_params_1", topic_name)
    utilities.send_file_kafka(kafka_producer, keys_dir, "tgsw_params_2", topic_name)
    utilities.send_file_kafka(kafka_producer, keys_dir, "boot_key_1", topic_name)
    utilities.send_file_kafka(kafka_producer, keys_dir, "boot_key_2", topic_name)

    logger.info("Shared keys to the aggregator")


def generate_he_keys(nb_classes: int, nb_teachers: int) -> None:
    """Generate HE keys and parameters

    Args:
        nb_classes (int): The number of classification classes
        nb_teachers (int): The number of PATE teachers
    """
    logger.info(
        "Generating HE keys for %s classes and %s teachers",
        str(nb_classes),
        str(nb_teachers),
    )
    fhe_dir: str = "/app_saferlearn/pate_he/"
    student_dir: str = fhe_dir + "student/"
    keys_dir: str = student_dir + "data/keys/"
    try:
        output: subprocess.CompletedProcess = subprocess.run(
            [
                student_dir + "bin/keyGen",
                keys_dir + "secret_key_1",
                keys_dir + "secret_key_2",
                keys_dir + "lwe_params_1",
                keys_dir + "lwe_params_2",
                keys_dir + "tgsw_params_1",
                keys_dir + "tgsw_params_2",
                keys_dir + "boot_key_1",
                keys_dir + "boot_key_2",
                keys_dir + "pub_key",
                keys_dir + "pub_params",
            ],
            capture_output=True,
            check=True,
        )
        utilities.log_subprocess(output)
    except subprocess.CalledProcessError as process_exception:
        utilities.log_subprocess(process_exception)
    share_he_keys()


def get_he_results(nb_classes: int) -> None:
    """Produces the result of the homomorphic aggregation

    Args:
        nb_classes (int): The number of classification classes
    """
    logger.debug("MODEL OWNER: get_he_results")
    topic_name: str = "model_he"
    group_id: str = "model_owner"
    consumer: KafkaConsumer = utilities.connect_kafka_consumer(
        topic_name, group_id, consumer_timeout_ms=1000
    )
    something_read: bool = False
    student_dir: Path = Path("/app_saferlearn/pate_he/student")
    student_data_dir: Path = student_dir / "data"
    student_ciphertext_dir: Path = student_data_dir / "ciphertexts"
    student_result_dir: Path = student_data_dir / "result"
    student_result_dir.mkdir(parents=True, exist_ok=True)

    # We should expect a certain number of results and wait for them
    # to be all transmitted
    while not something_read:
        time.sleep(1)
        for msg in consumer:
            vote_index: str = (msg.key).decode("utf-8")
            data: bytes = msg.value
            logger.debug("vote %s", vote_index)
            student_ciphertext_vote_dir: Path = student_ciphertext_dir / vote_index
            try:
                student_ciphertext_vote_dir.mkdir(parents=True, exist_ok=True)
            except OSError as error:
                logger.error("Error while creating directory: %s", error)
            vote_file: Path = student_ciphertext_vote_dir / "encrypted_argmax"
            with open(vote_file, "wb") as encrypted_result_file:
                encrypted_result_file.write(data)
            something_read = True

    consumer.close()
    utilities.clean_topic(topic_name)

    encrypted_results: List[Path] = [
        result for result in student_ciphertext_dir.iterdir() if result.is_dir()
    ]

    # This is the starting point of the decryption
    decryption_start_time = time.time()

    for encrypted_result in encrypted_results:
        logger.info("Proceeding to decryption")
        try:
            output: subprocess.CompletedProcess = subprocess.run(
                [
                    student_dir / "bin/decrypt",
                    encrypted_result / "encrypted_argmax",
                    student_data_dir / "keys/lwe_params_1",
                    str(nb_classes),
                    student_data_dir / "keys/secret_key_1",
                    student_result_dir / f"{encrypted_result.stem}_plain_argmax",
                ],
                capture_output=True,
                check=True,
            )
            utilities.log_subprocess(output)
        except subprocess.CalledProcessError as process_exception:
            utilities.log_subprocess(process_exception)

    # The decryption is over
    decryption_end_time = time.time()

    decryption_time = decryption_end_time - decryption_start_time
    logger.info("Decryption took %s s", decryption_time)


def get_results() -> None:
    """Get clear results"""
    logger.debug("MODEL OWNER: get_results")
    topic_name: str = "model_pate"
    group_id: str = "model_owner"
    consumer: KafkaConsumer = utilities.connect_kafka_consumer(
        topic_name, group_id, consumer_timeout_ms=1000
    )
    something_read: bool = False
    student_dir: Path = Path("/app_saferlearn/pate_he/student")
    student_data_dir: Path = student_dir / "data"
    student_result_dir: Path = student_data_dir / "result"
    student_result_dir.mkdir(parents=True, exist_ok=True)

    try:
        student_result_dir.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        logger.error("Error while creating directory: %s", error)
    vote_file: Path = student_result_dir / "pate_clear.csv"

    # We should expect a certain number of results and wait for them
    # to be all transmitted
    with open(vote_file, "w", encoding="utf-8") as result_file:
        while not something_read:
            time.sleep(1)
            for msg in consumer:
                vote_index: str = (msg.key).decode("utf-8")
                data: bytes = msg.value.decode("utf-8")
                logger.debug("vote %s : %s", vote_index, data)
                result_file.write(f"{vote_index},{data}\n")
                something_read = True

    consumer.close()
    utilities.clean_topic(topic_name)


def get_mpc_results() -> None:
    """Produce the result of the multi-party aggregation"""
    logger.debug("MODEL OWNER: get_mpc_results")
    topic_name: str = "model_mpc"
    group_id: str = "model_owner"
    consumer: KafkaConsumer = utilities.connect_kafka_consumer(
        topic_name, group_id, consumer_timeout_ms=0
    )
    # Right now, we de not receive any result by Kafka
    for msg in consumer:
        vote_index: int = int((msg.key).decode("utf-8"))
        logger.debug(
            "MPC vote for index %s : %s", vote_index, msg.value.decode("utf-8")
        )
    consumer.close()
    utilities.clean_topic(topic_name)


def main() -> None:
    """Main entry for the student"""
    logger.info("Student started")
    params: Dict[str, str] = retrieve_params()
    for key, value in params.items():
        logger.debug("Received %s : %s", key, value)
    algorithm: str = params["algorithm"]

    # This is the starting point of the computation
    start_time = time.time()

    if algorithm == "mpc":
        get_mpc_results()
    elif algorithm == "pate":
        get_results()
    elif algorithm == "fhe":
        generate_he_keys(int(params["nb_classes"]), int(params["nb_teachers"]))
        get_he_results(int(params["nb_classes"]))
    elif algorithm == "fl":
        logger.error("Federated learning is not yet implemented, defaulting to HE")
        get_he_results(int(params["nb_classes"]))
    else:
        logger.error("Incorrect algorithm provided by the orchestrator : %s", algorithm)
    logger.debug("MODEL OWNER: votes retrieved")

    # The computation is over
    end_time = time.time()

    overall_computation_time = end_time - start_time
    logger.info(
        "Overall computation for : %s took %s s", algorithm, overall_computation_time
    )


if __name__ == "__main__":
    main()
