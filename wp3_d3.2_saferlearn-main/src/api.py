#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File              : api.py
# Author            : hargathor <3949704+hargathor@users.noreply.github.com>
# Date              : 02.04.2019
# Last Modified Date: 29.04.2022
# Last Modified By  : hargathor <3949704+hargathor@users.noreply.github.com>

"""API module"""

import json
import logging
import threading
from pathlib import Path
import torchvision
from typing import Any, Dict, List, Optional

from flask import Flask, request
from flask.wrappers import Response
from flask_cors import CORS

import db_utilities
import utilities
from config import Config
from orchestrator import Job
from huggingface_hub import list_datasets
from typeguard import typechecked
import typer
from typing_extensions import Annotated
import torchvision


# Directory path where model should be (TO CHANGE)
# modelsPath: Path = Path("/tmp/trained_nets")

app: Flask = Flask(__name__)
app.config.from_object(Config)
CORS(app)

logging.basicConfig(level=logging.DEBUG)
logger: logging.Logger = logging.getLogger("api")
logger.setLevel(logging.DEBUG)

logging.getLogger("requests").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.INFO)
logging.getLogger("kafka").setLevel(logging.WARNING)
logging.getLogger("werkzeug").setLevel(logging.WARNING)
logging.getLogger("watchdog").setLevel(logging.WARNING)


@app.route("/job", methods=["PUT"])
def api_start_job() -> Response:
    """Starts a Saferlearn job when receiving a request from the UI.

    Returns:
        Response: The HTTP response
    """
    logger.info("Starting job from request")
    body: Optional[Any] = request.get_json()
    if body is not None:
        algorithm: str = body["algorithm"]
        datatype: str = body["datatype"]
        dataset: str = body["dataset"]
        workers_list: List[Any] = body["workers"]
        nb_classes: int = int(body["nbClasses"])
        dp_value: float = float(body["dpValue"])
        use_differential_privacy: bool = body["useDP"]
        if not use_differential_privacy:
            dp_value = 0.0
        # mimick behavior for now
        # nb_samples = body["nb_samples"]
        nb_samples = 1000
    else:
        return Response(
            "No parameters received",
            status=405,
            mimetype="application/json",
        )
    # global NB_COMPUTING_PARTIES
    nb_computing_parties: int = app.config.get(
        "NB_COMPUTING_PARTIES"
    )  # TODO: pass it via gui
    active_workers: List[Dict[str, Any]] = db_utilities.get_active_workers(
        datatype=datatype, state="online"
    )
    workers: List[Dict[str, Any]] = active_workers[: len(workers_list)]
    if len(workers) > 0:
        current_job: Job = Job(workers, nb_classes, algorithm, datatype, dataset)
        db_utilities.register_job(current_job.uuid, workers, datatype)
        if algorithm == "mpc":
            active_computing_parties: List[Dict[str, Any]] = (
                db_utilities.get_active_workers(
                    datatype="COMPUTING_PARTY", state="online"
                )
            )
            computing_parties: List[Dict[str, Any]] = active_computing_parties[
                0:nb_computing_parties
            ]
            logger.debug("Computing parties: %s", computing_parties)
            thread: threading.Thread = threading.Thread(
                target=current_job.create_job_mpc,
                args=(computing_parties,),
            )
        elif algorithm == "fl":
            # TODO: implement a flower connector/job
            logging.error("Not Implemented")
            response: Response = Response(
                "Federated Learning not implemented",
                status=400,
                mimetype="application/text",
            )
        elif algorithm == "pate":
            logger.debug("PATE")
            thread = threading.Thread(
                target=current_job.create_job_clear,
                args=(
                    nb_samples,
                    dp_value,
                ),
            )
        else:  # default is fhe
            logger.debug("PATE with FHE")
            thread = threading.Thread(
                target=current_job.create_job,
                args=(
                    nb_samples,
                    dp_value,
                ),
            )
        thread.start()
        response: Response = Response(
            json.dumps(current_job.__dict__), status=200, mimetype="application/json"
        )
    else:
        response: Response = Response(
            "No workers affected to job", status=405, mimetype="application/json"
        )

    return response


@app.route("/datasets", methods=["GET"])
def get_datasets() -> Response:
    """Provide the list of available datasets.

    Returns:
        Response: The list of datasets
    """
    path_datasets: Path = Path(app.config.get("DATASETS_PATH"))
    datasets: List[str] = [
        dataset.stem for dataset in path_datasets.iterdir() if dataset.is_dir()
    ]
    datasets_json: Dict[str, List[str]] = {"datasets": datasets}
    # datasets_list = list_datasets(limit=10)
    # datasets_list = torchvision.datasets.__all__
    # datasets: List[str] = []
    # for dataset in datasets_list:
    #     datasets.append(dataset)
    # # logger.debug(list(datasets_list))
    # for dataset in datasets_list:
    #     datasets.append(
    #         {
    #             "id": dataset.id,
    #             "author": dataset.author,
    #             "tags": dataset.tags,
    #         }
    #     )
    datasets_json: Dict[str, List[str]] = {"datasets": datasets}
    logger.debug(json.dumps(datasets_json))

    return Response(json.dumps(datasets_json), status=200, mimetype="application/json")


@app.route("/job/<job_uuid>/<state>", methods=["PUT"])
def update_job(job_uuid: str, state: str) -> Response:
    """Change job's state

    Args:
        job_uuid (str): The job UUID
        state (str): The new state

    Returns:
        Response: The new job
    """
    db_utilities.update_job_state(job_uuid, state)
    return Response(
        json.dumps(db_utilities.get_local_job(job_uuid)),
        status=200,
        mimetype="application/json",
    )


@app.route("/job/<job_id>", methods=["GET"])
def get_job(job_id: str) -> Response:
    """Provide information about a specific job

    Args:
        job_id (str): UUID of the job

    Returns:
        Response: The json representation of a Job
    """
    return Response(
        json.dumps(db_utilities.get_local_job(job_id)),
        status=200,
        mimetype="application/json",
    )


@app.route("/jobs", methods=["GET"])
def get_jobs() -> Response:
    """Retrieve the list of jobs in DB

    Returns:
        Response: The list of jobs
    """
    return Response(
        json.dumps(db_utilities.get_local_jobs()),
        status=200,
        mimetype="application/json",
    )


@app.route("/jobs", methods=["DELETE"])
def flush_jobs() -> Response:
    """Remove all jobs from DB

    Returns:
        Response: The HTTP response
    """
    db_utilities.delete_table("jobs")
    return Response(status=200, mimetype="application/json")


@app.route("/clients", methods=["GET"])
def get_workers() -> Response:
    """Retrieve the list of active workers in DB

    Returns:
        Response: The list of all workers
    """
    return Response(
        json.dumps(db_utilities.get_active_workers()),
        status=200,
        mimetype="application/json",
    )


@app.route("/clients", methods=["DELETE"])
def flush_workers() -> Response:
    """Remove all workers from DB

    Returns:
        Response: The HTTP response
    """
    db_utilities.delete_table("workers")
    return Response(status=200, mimetype="application/json")


@app.route("/client/<worker_id>/job/<job_uuid>/finish", methods=["PUT"])
def finish_job(worker_id: str, job_uuid: str) -> Response:
    """Set a specific job as finished

    Args:
        worker_id (str): The worker's UUID
        job_uuid (str): The job's UUID

    Returns:
        Response: The HTTP response
    """
    logger.info("Worker %s has finished for job %s", worker_id, job_uuid)
    db_utilities.update_job_worker_state(job_uuid, "done", worker_id)
    return Response(status=200, mimetype="application/json")


@app.route("/client/<worker_id>", methods=["PUT"])
def register_worker(worker_id: str) -> Response:
    """Updating a worker

    Args:
        worker_id (UUID): The worker's UUID

    Returns:
        Response: The HTTP response
    """
    worker: Dict[str, Any] = {"id": worker_id}
    if "ip" in request.headers:
        worker["ip"] = request.headers["ip"]
    if "port" in request.headers:
        worker["port"] = int(request.headers["port"])
    if "data" in request.headers:
        worker["data_format"] = request.headers["data"]
    worker_stringified: str = json.dumps(worker)
    # logger.debug(f"Received a registration request from worker {worker_stringified}")
    db_utilities.update_db_worker(worker)
    return Response(worker_stringified, status=200, mimetype="application/json")


def main(
    datasets_path: Annotated[
        str,
        typer.Option(
            help="Public datasets path",
        ),
    ],
) -> None:
    app.config.update(DATASETS_PATH=datasets_path)
    app.run(host=app.config["BIND_HOST"])
    # app.run(debug=app.config["DEBUG"], host=app.config["BIND_HOST"])


if __name__ == "__main__":
    # utilities.release_model_dir(modelsPath)
    typer.run(main)
