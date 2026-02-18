#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File              : data_owner.py
# Author            : hargathor <3949704+hargathor@users.noreply.github.com>
# Date              : 09.04.2019
# Last Modified Date: 09.04.2019
# Last Modified By  : hargathor <3949704+hargathor@users.noreply.github.com>

"""Computing party module"""

import argparse
import logging
import os
import socket
import subprocess
import threading
from signal import SIGTERM
from typing import Any, Dict, List, Optional

import rpyc
from psutil import process_iter
from rpyc.utils import server

import utilities

logging.basicConfig(level=logging.DEBUG)
logger: logging.Logger = logging.getLogger("computing_party")
logging.getLogger("urllib3").setLevel(logging.WARNING)


RPYC_PORT: Optional[str] = os.environ.get("RPYC_PORT")
RPYC_HOST: str = socket.gethostbyname(socket.gethostname())

ORCHESTRATOR_HOST: Optional[str] = os.environ.get("ORCHESTRATOR_HOST")


class WorkerService(rpyc.Service):
    """Remote Python Call service

    Args:
        rpyc (_type_): _description_

    Returns:
        _type_: _description_
    """

    state: str = "waiting"

    def __init__(
        self,
        worker_uuid: str,
        aggregator_id: int,
        number_of_aggregators: int,
        mpc_program: str,
        mpc_protocol: str,
        tag: str,
    ) -> None:
        self.worker_uuid: str = worker_uuid
        self.tag: str = tag
        self.aggregator_id: int = aggregator_id
        self.number_of_aggregators: int = number_of_aggregators
        self.mpc_program: str = mpc_program
        self.mpc_protocol: str = mpc_protocol

    def __call__(self):
        super().__init__()
        return self

    def exposed_get_state(self) -> str:
        """exposes the state of the computing party

        Returns:
            _type_: _description_
        """
        return self.state

    def on_connect(self, conn) -> None:
        """code that runs when a connection is created
        (to init the service, if needed)

        Args:
            conn (_type_): _description_
        """

    def on_disconnect(self, conn) -> None:
        """code that runs after the connection has already closed
        (to finalize the service, if needed)

        Args:
            conn (_type_): _description_
        """

    def exposed_launch_computing_party(
        self, hosts: List[Dict[str, Any]], nb_workers: int
    ) -> None:
        """Hook to launch the MPC computation

        Args:
            hosts (List[Dict[str, Any]]): _description_
            nb_workers (int): _description_
        """
        mpc_base_path = "/MPC"
        if not os.path.exists(mpc_base_path):
            os.mkdir(mpc_base_path)
        os.chdir(mpc_base_path)
        # here kill the process that use the ports for MPC
        for proc in process_iter():
            for conns in proc.connections(kind="inet"):
                if (
                    (conns.laddr.port == 14000)
                    or (conns.laddr.port == 14001)
                    or (conns.laddr.port == 14002)
                ):
                    proc.send_signal(SIGTERM)  # or SIGKILL
                # FIXME: ports are hardcoded
        aggregator_id_number: int = 0
        with open("/app_saferlearn/HOSTS", "w", encoding="utf-8") as host_file:
            counter: int = 0
            for host in hosts:
                host_file.write(str(host["ip"]) + "\n")
                if host["ip"] == RPYC_HOST:
                    aggregator_id_number = counter
                counter += 1
        try:
            compile_subprocess: subprocess.CompletedProcess = subprocess.run(
                [
                    "python3",
                    "/MPC/compile.py",
                    f"/MPC/Programs/Source/{self.mpc_program}",
                    "--batch-size",
                    "10",
                    "--max-num-teachers",
                    str(nb_workers),
                ],
                capture_output=True,
                check=True,
            )
            utilities.log_subprocess(compile_subprocess)
        except subprocess.CalledProcessError as process_exception:
            utilities.log_subprocess(process_exception)
        try:
            mpc_subprocess: subprocess.CompletedProcess = subprocess.run(
                [
                    f"/MPC/{self.mpc_protocol}-party.x",
                    "-p",
                    str(aggregator_id_number),
                    "-N",
                    str(len(hosts)),
                    "--ip-file-name",
                    "/app_saferlearn/HOSTS",
                    self.mpc_program,
                ],
                capture_output=True,
                check=True,
            )
            utilities.log_subprocess(mpc_subprocess)
        except subprocess.CalledProcessError as process_exception:
            utilities.log_subprocess(process_exception)


def launch_rpyc_server(
    hostname: str,
    port: str,
    worker_uuid: str,
    auto_register: bool,
    protocol_config: Dict[str, bool],
    backlog: int,
    aggregator_id: int,
    number_of_aggregators: int,
    mpc_program: str,
    mpc_protocol: str,
    tag: str,
) -> None:
    """_summary_

    Args:
        hostname (str): _description_
        port (str): _description_
        worker_uuid (str): _description_
        auto_register (bool): _description_
        protocol_config (Dict[str, bool]): _description_
        backlog (int): _description_
        aggregator_id (int): _description_
        number_of_aggregators (int): _description_
        mpc_program (str): _description_
        mpc_protocol (str): _description_
        tag (str): _description_
    """
    server.ThreadedServer(
        WorkerService(
            worker_uuid,
            aggregator_id,
            number_of_aggregators,
            mpc_program,
            mpc_protocol,
            tag,
        ),
        hostname=hostname,
        port=int(port),
        auto_register=auto_register,
        protocol_config=protocol_config,
        backlog=backlog,
    ).start()


def main() -> None:
    """_summary_"""
    parser: argparse.ArgumentParser = argparse.ArgumentParser()
    parser.add_argument(
        "-t",
        "--tag",
        default="COMPUTING_PARTY",
        help="Node type, default 'COMPUTING_PARTY'",
    )
    parser.add_argument(
        "-p", "--computing-party-id", default="0", help="computing party ID, default 0"
    )
    parser.add_argument(
        "-N",
        "--computing-parties-number",
        default="3",
        help="Number of computing parties, default 3",
    )
    parser.add_argument(
        "--protocol",
        default="mascot",
        help="Protocol for secure multi-party computation",
    )
    parser.add_argument(
        "program", default="pate_aggregation", help="Program to run on the aggregators"
    )
    args: argparse.Namespace = parser.parse_args()

    tag: str = args.tag
    aggregator_id: int = args.computing_party_id
    number_of_aggregators: int = args.computing_parties_number
    mpc_program: str = args.program
    mpc_protocol: str = args.protocol

    protocol_config: Dict[str, bool] = dict(
        instantiate_custom_exceptions=True, import_custom_exceptions=True
    )

    logger.info("Generating UUID")
    worker_uuid: str = utilities.generate_uuid()
    logger.info("Launching Computing Party")
    threads: List[threading.Thread] = []
    thread_computing: threading.Thread = threading.Thread(
        target=launch_rpyc_server,
        args=(
            RPYC_HOST,
            RPYC_PORT,
            worker_uuid,
            False,
            protocol_config,
            500,
            aggregator_id,
            number_of_aggregators,
            mpc_program,
            mpc_protocol,
            tag,
        ),
    )
    logger.debug("Starting data owner with %s datatype", tag)
    thread_heartbeat: threading.Thread = threading.Thread(
        target=utilities.heartbeat,
        args=(
            RPYC_HOST,
            RPYC_PORT,
            worker_uuid,
            tag,
            5,
        ),
    )
    threads.append(thread_computing)
    threads.append(thread_heartbeat)
    thread_computing.start()

    logger.debug("Registering worker")
    thread_heartbeat.start()


if __name__ == "__main__":
    main()
