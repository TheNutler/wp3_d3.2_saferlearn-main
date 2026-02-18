"""Utilities module containing various common functions"""

import asyncio
import json
import logging
import os
import pickle
from subprocess import CompletedProcess, CalledProcessError
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from config import Config

import numpy as np
import requests
from kafka import KafkaConsumer, KafkaProducer
from kafka.admin import KafkaAdminClient
from kafka.errors import KafkaError, UnknownTopicOrPartitionError
from requests import Response

ORCHESTRATOR_HOST: Optional[str] = os.environ.get("ORCHESTRATOR_HOST")
ORCHESTRATOR_PORT: Optional[int] = os.environ.get("ORCHESTRATOR_PORT")

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("utilities")
logger.setLevel(logging.DEBUG)

local_config = Config()


def publish_message(
    producer_instance: KafkaProducer, topic_name: str, key, value, encode=True
) -> None:
    """_summary_

    Args:
        producer_instance (KafkaProducer): _description_
        topic_name (str): _description_
        key (_type_): _description_
        value (_type_): _description_
        encode (bool, optional): _description_. Defaults to True.
    """
    try:
        key_bytes: bytes = bytes(str(key), encoding="utf-8")
        if encode:
            value_bytes: bytes = bytes(str(value), encoding="utf-8")
        else:
            value_bytes = value
        future = producer_instance.send(topic_name, key=key_bytes, value=value_bytes)
        future.get(timeout=10)
        # logger.info("Published %s bytes on %s", sys.getsizeof(value_bytes), topic_name)
        # logger.debug("Key is %s", key)
    except KafkaError as ex:
        logger.error("Exception while publishing message")
        logger.error(str(ex))


def connect_kafka_consumer(
    topic_name: str,
    group_id: str,
    consumer_timeout_ms: int = 305000,
    enable_auto_commit: bool = True,
) -> KafkaConsumer:
    """Create a Kafka consumer instance

    Args:
        topic_name (str, optional): Name of the topic. Defaults to "votes_he".
        group_id (str, optional): Group id for concurrent reads.
        consumer_timeout_ms (int, optional): Timeout. Defaults to 305000.
        enable_auto_commit (boolean, optional): See kafka-python doc. \
            Defaults to=True.

    Returns:
        KafkaConsumer: The consumer instance
    """
    _consumer: KafkaConsumer = KafkaConsumer(
        topic_name,
        auto_offset_reset="earliest",
        bootstrap_servers=[f"{local_config.KAFKA_HOST}:{local_config.KAFKA_PORT}"],
        api_version=(0, 10),
        max_partition_fetch_bytes=2000000000,
        enable_auto_commit=enable_auto_commit,
        consumer_timeout_ms=consumer_timeout_ms,
        # group_id=group_id,
    )
    return _consumer


def clean_topic(topic_name: str) -> None:
    """Clean the topic

    Args:
        topic_name (str): topic name to clean
    """
    admin: KafkaAdminClient = KafkaAdminClient(
        bootstrap_servers=[f"{local_config.KAFKA_HOST}:{local_config.KAFKA_PORT}"]
    )
    try:
        admin.delete_topics([topic_name])
    except UnknownTopicOrPartitionError:
        logger.error("Unknown Topic")


def connect_kafka_producer() -> KafkaProducer:
    """Create a Kafka producer instance

    Returns:
        KafkaProducer: The producer instance
    """
    _producer: KafkaProducer = KafkaProducer(
        bootstrap_servers=[f"{local_config.KAFKA_HOST}:{local_config.KAFKA_PORT}"],
        api_version=(0, 10),
    )
    return _producer
    # try:
    # except KafkaError as ex:
    #     logger.error("Exception while connecting kafka")
    #     logger.error(str(ex))


def get_list_of_workers(job_uuid: str) -> List[str]:
    """List workers linked to a job

    Args:
        job_uuid (str): ID of job

    Returns:
        List[str]: The list of workers working on the job
    """
    url: str = f"http://{ORCHESTRATOR_HOST}:{ORCHESTRATOR_PORT}/job/{job_uuid}"
    logger.debug("Fetching workers on %s", url)

    try:
        res: Response = requests.get(url, timeout=10)
        if res.status_code != 200:
            logger.error("HTTP error: %s", res.json())
            return []
        else:
            return res.json()["workers"]
    except ConnectionError as err:
        logger.error("Connection error: %s", err)
        return []


def release_model_dir(home_path: Path) -> None:
    """Release locks of model directories

    Args:
        homePath (Path): Path of model directories
    """
    if home_path.exists() is False:
        raise FileNotFoundError("Invalid path : {home_path}")
    for model in [model for model in home_path.iterdir() if model.is_dir()]:
        lock_file_name: Path = model / ".lock"
        if lock_file_name.exists():
            lock_file_name.unlink()
            print(model, " lock removed")


def register_finish(
    broker_addr: str,
    broker_port: str,
    worker_uuid: str,
    job_uuid: str,
    rpyc_host: str,
    rpyc_port: str,
) -> None:
    """Tells the api the job is done

    Args:
        broker_addr (str): The broker address
        broker_port (str): The broker port
        worker_uuid (str): UUID of the worker
        job_uuid (str): UUID of the job
        rpyc_host (str): _description_
        rpyc_port (str): _description_
    """
    url: str = (
        f"http://{broker_addr}:{broker_port}/client/{worker_uuid}/job/{job_uuid}/finish"
    )

    headers: Dict[str, Any] = {"ip": rpyc_host, "port": str(rpyc_port)}
    logger.debug(url)  # add context
    logger.debug(headers)  # add context
    try:
        res: Response = requests.put(url, headers=headers, timeout=60)
        if res.status_code != 200:
            logger.error("HTTP error: %s", res.json())
    except ConnectionError as err:
        logger.error("Connection error: %s", err)


def registering(
    broker_addr: str,
    broker_port: str,
    worker_uuid: str,
    worker_data_format: str,
    rpyc_host: str,
    rpyc_port: str,
) -> None:
    """Register a worker

    Args:
        broker_addr (str): The broker address
        broker_port (str): The broker port
        worker_uuid (str): UUID of the worker registering
        worker_data_format (str): The data handled by the worker
        rpyc_host (str): IP of the worker registering
        rpyc_port (str): Port of the worker registering
    """
    url: str = f"http://{broker_addr}:{broker_port}/client/{worker_uuid}"
    headers: Dict[str, Any] = {
        "ip": rpyc_host,
        "port": str(rpyc_port),
        "data": worker_data_format,
    }
    # logger.debug(f"Registering worker {worker_uuid} with headers {headers}")
    try:
        req: Response = requests.put(url, headers=headers, timeout=60)
        if req.status_code != 200:
            logger.error("HTTP error: %s", req.json())
    except requests.exceptions.ConnectionError as err:
        logger.error("Connection error: %s", err)
        logger.error("The API doesn't seems to be listening")


def heartbeat(
    orchestrator_host: str,
    orchestrator_port: str,
    rpyc_host: str,
    rpyc_port: str,
    worker_uuid: str,
    data_format: str,
    seconds: int = 60,
) -> None:
    while True:
        if orchestrator_host is not None:
            registering(
                orchestrator_host
                if orchestrator_host is not None
                else ORCHESTRATOR_HOST,
                orchestrator_port
                if orchestrator_port is not None
                else ORCHESTRATOR_PORT,
                worker_uuid,
                data_format,
                rpyc_host,
                rpyc_port,
            )
        time.sleep(seconds)


def generate_uuid() -> str:
    """Generate a Universally Unique Identifier

    Returns:
        str: The identifier
    """
    worker_uuid: uuid.UUID = uuid.uuid4()
    # so multiple workers on the same hosts are an issue ?
    # worker_path: Path = Path("/tmp/worker")
    # if not worker_path.exists():
    #     logger.debug("Non existing uuid, generating one...")
    #     worker_uuid: uuid.UUID = uuid.uuid4()
    #     with open(worker_path, "wb") as worker_file:
    #         pickle.dump(worker_uuid, worker_file)
    # else:
    #     logger.debug("Loading existing UUID")
    #     with open(worker_path, "rb") as worker_file:
    #         worker_uuid = pickle.load(worker_file)

    # logger.debug("UUID is %s", worker_uuid)

    return str(worker_uuid)


def log_subprocess(
    process: Union[CompletedProcess, CalledProcessError],
) -> None:
    """Human friendly subprocess log

    Args:
        subprocess (subprocess): The return value of the subprocess to log
    """
    if isinstance(process, CompletedProcess):
        logger.info(" ".join(map(str, process.args)))
    else:
        logger.error(" ".join(map(str, process.cmd)))
    logger.debug("stdout : %s", process.stdout.decode("utf-8"))
    logger.debug("stderr : %s", process.stderr.decode("utf-8"))
    logger.info("returncode : %s", str(process.returncode))


def send_file_kafka(producer: KafkaProducer, path: Path, name: str, topic: str) -> None:
    """Send a file through Kafka chunk by chunk

    Args:
        producer (KafkaProducer): The Kafka producer to use
        path (Path): The folder containing the file to send
        name (str): The name of the file
        topic (str): The Kafka topic to use
    """
    nb_chunks: int = 0
    with open(path / name, "rb") as file_to_send:
        while True:
            chunk: bytes = file_to_send.read(
                1000 * 1024  # Accounts for the serialization overhead
            )
            if not chunk:
                logger.debug("Sent %s file in %s chunks", name, nb_chunks)
                publish_message(
                    producer,
                    topic,
                    '{"file": "' + name + '", "chunk_id": "-1"}',
                    str(nb_chunks),
                )
                return
            publish_message(
                producer,
                topic,
                '{"file": "' + name + '", "chunk_id": "' + str(nb_chunks) + '"}',
                chunk,
                encode=False,
            )

            nb_chunks += 1


def receive_file_kafka(consumer: KafkaConsumer, path: Path, name: str) -> None:
    """Receive a file through Kafka chunk by chunk

    Args:
        consumer (KafkaConsumer): The Kafka consumer to use
        path (Path): The destination folder
        name (str): The name of the file
    """
    logger.debug("Waiting to receive %s", name)
    nb_chunks: int = 0
    with open(path / name, "wb") as file_to_receive:
        for msg in consumer:
            key: Dict[str, Any] = json.loads((msg.key).decode("utf-8"))
            if key["file"] == name:
                if int(key["chunk_id"]) == -1:
                    nb_chunks_verification: int = int(msg.value.decode("utf-8"))
                    if nb_chunks_verification != nb_chunks:
                        logger.error(
                            "Received %s chunks of %s but needed %s",
                            nb_chunks,
                            name,
                            nb_chunks_verification,
                        )
                    else:
                        logger.debug("Received %s file in %s chunks", name, nb_chunks)
                    break
                elif int(key["chunk_id"]) == nb_chunks:
                    chunk: bytes = msg.value
                    file_to_receive.write(chunk)
                    nb_chunks += 1
                else:
                    logger.error(
                        "Received chunk %s, expected chunk %s",
                        key["chunk_id"],
                        nb_chunks,
                    )
            else:
                logger.error("Received key %s while expecting %s", key["file"], name)
