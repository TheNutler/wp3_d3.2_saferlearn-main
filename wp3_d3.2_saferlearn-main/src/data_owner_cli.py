#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File              : data_owner.py
# Author            : hargathor <3949704+hargathor@users.noreply.github.com>
# Date              : 09.04.2019
# Last Modified Date: 29.04.2022
# Last Modified By  : hargathor <3949704+hargathor@users.noreply.github.com>

"""Data owner module"""

import argparse
import logging as logger
import os
import socket
import subprocess
import threading
from enum import Enum
from typing import List, Optional

from usecases.data_owner_example import ThalesDataOwner

import utilities
from usecases.data_owner_abstract_class import DataOwner
from saferlearn import MPCParameters

from typeguard import typechecked
import typer
from typing_extensions import Annotated

logger.basicConfig(level=logger.DEBUG)
logger.getLogger("requests").setLevel(logger.WARNING)
logger.getLogger("urllib3").setLevel(logger.WARNING)
logger.getLogger("kafka").setLevel(logger.WARNING)
logging: logger.Logger = logger.getLogger(__name__)

RPYC_PORT: Optional[str] = os.environ.get("RPYC_PORT")
RPYC_HOST: str = socket.gethostbyname(socket.gethostname())
ORCHESTRATOR_HOST: Optional[str] = os.environ.get("ORCHESTRATOR_HOST")


class UseCase(Enum):
    Stub = ThalesDataOwner

    def __str__(self) -> str:
        return self.name


@typechecked
def main_typer(
    type: Annotated[
        str,
        typer.Option(
            default_factory=UseCase.Stub.name,
            help=f"Node type, default '{UseCase.Stub.name}'",
        ),
    ],
    program: Annotated[
        str,
        typer.Argument(
            help="Client program to run by the teachers",
        ),
    ] = ("pate-teacher"),
    computing_parties_number: Annotated[
        int, typer.Option(help="Number of computing parties")
    ] = 3,
    nteachers: Annotated[int, typer.Option(help="Number of teachers")] = 3,
    size_batch: Annotated[
        int,
        typer.Option(
            help="Size of batch",
        ),
    ] = 10,
    sigma: Annotated[
        int,
        typer.Option(
            help="Sigma for Gaussian noise distribution",
        ),
    ] = 1,
    rounds: Annotated[
        int,
        typer.Option(
            help="Number of rounds for noise",
        ),
    ] = 1,
    data_format: Annotated[
        str,
        typer.Option(
            help="Your type of data",
        ),
    ] = "radio",
    rpyc_host: Annotated[
        str,
        typer.Option(
            help="Your advertised listening address",
        ),
    ] = RPYC_HOST,
    rpyc_port: Annotated[
        int,
        typer.Option(
            help="Your listening port",
        ),
    ] = 1244,
    orchestrator_host: Annotated[
        str,
        typer.Option(
            help="Orchestrator ip address or hostname",
        ),
    ] = "127.0.0.1",
    orchestrator_port: Annotated[
        int,
        typer.Option(
            help="Orchestrator listening port",
        ),
    ] = 5000,
) -> None:
    tag: str = data_format
    use_case_cls = UseCase[type].value

    logger.info("Generating UUID")
    worker_uid: str = utilities.generate_uuid()
    logger.debug("Launching Worker RPyC")

    mpc_parameters: MPCParameters = MPCParameters(
        computing_parties_number,
        size_batch,
        program,
        sigma,
        rounds,
    )
    data_owner: DataOwner = use_case_cls(
        worker_uid,
        nteachers,
        tag,
        mpc_parameters=mpc_parameters,
        rpyc_host=rpyc_host,
        rpyc_port=rpyc_port,
    )

    threads: List[threading.Thread] = []

    teacher_subprocess: threading.Thread = threading.Thread(
        target=data_owner.launch_server
    )

    threads.append(teacher_subprocess)

    logger.debug("Starting data owner with %s datatype", tag)
    heartbeat_subprocess: threading.Thread = threading.Thread(
        target=utilities.heartbeat,
        args=(
            orchestrator_host,
            orchestrator_port,
            rpyc_host,
            rpyc_port,
            worker_uid,
            tag,
            5,
        ),
    )
    threads.append(heartbeat_subprocess)
    teacher_subprocess.start()

    logger.debug("Registering worker")
    heartbeat_subprocess.start()


if __name__ == "__main__":
    typer.run(main_typer)
