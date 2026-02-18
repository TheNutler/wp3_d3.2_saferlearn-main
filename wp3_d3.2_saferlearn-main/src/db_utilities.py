"""DB utilities module"""

import logging
import sqlite3 as lite
from sqlite3 import Connection, Cursor
from typing import Any, Dict, List, Optional, Tuple
from config import Config

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("db_utilities")
logger.setLevel(logging.WARNING)
local_config = Config()


def connect_to_db(development: bool = False) -> Connection:
    """Connect to the DB

    Args:
        development (bool, optional): _description_. Defaults to False.

    Returns:
        Connection: The DB connection
    """

    if development:
        connection: Connection = lite.connect(":memory:")
    else:
        connection: Connection = lite.connect(
            local_config.WORKER_DB_PATH, check_same_thread=False
        )
    try:
        with connection:
            cur: Cursor = connection.cursor()
            cur.execute(
                "CREATE TABLE IF NOT EXISTS workers(id TEXT, ip TEXT, port TEXT,"
                " datatype TEXT, date DATE, state TEXT, worker_number TEXT)"
            )
            cur.execute(
                "CREATE TABLE IF NOT EXISTS"
                " jobs(id TEXT, datatype TEXT, date DATE, state TEXT, worker_id TEXT)"
            )
    except lite.Error as err:
        logger.error("Error in connect to DB: %s", err.args[0])

    return connection


# FIXME: update_db seems too vague while we only set workers as online
def update_db_worker(worker: Dict[str, Any]) -> None:
    """Set worker as online

    Args:
        worker (Dict[str, Any]): _description_
    """
    logger.debug("Updating worker %s", worker)
    update_state_workers()
    connection: Optional[Connection] = None
    worker_number: int = get_worker_number(worker["id"])
    worker_state: str = "online"
    try:
        connection = connect_to_db(development=False)
        with connection:
            cur: Cursor = connection.cursor()
            cur.execute(f"DELETE FROM workers WHERE id='{worker['id']}'")
            request: str = (
                "INSERT INTO workers VALUES("
                "'{}', '{}', '{}', '{}', datetime('now'), '{}', '{}'"
                ")"
            ).format(
                worker["id"],
                worker["ip"],
                worker["port"],
                worker["data_format"],
                worker_state,
                worker_number,
            )
            cur.execute(request)
            logger.debug("Worker %s set online", worker["id"])
    except lite.Error as err:
        logger.error("Error in update DB worker: %s", err.args[0])
    finally:
        if connection is not None:
            connection.close()


def update_state_workers() -> None:
    """Set old workers are dead in DB"""
    # TODO: This is probably too naive
    connection: Optional[Connection] = None
    try:
        connection = connect_to_db(development=False)
        with connection:
            cur: Cursor = connection.cursor()
            cur.execute(
                "UPDATE workers SET state = 'dead'"
                " WHERE datetime('now', '-10 second') > date"
            )
            cur.execute(
                "DELETE FROM workers WHERE state = 'dead' AND datetime('now', '-60 second') > date"
            )
    except lite.Error as err:
        logger.error("Error in update state workers: %s", err.args[0])
    finally:
        if connection is not None:
            connection.close()


def get_worker_number(worker_uuid: str) -> int:
    """Retrieve the number of worker by its UUID

    Args:
        worker_uuid (str): The UUID of the worker

    Returns:
        int: The worker number
    """
    # TODO: What is the worker_number ?
    logger.debug("Fetching worker number for id %s", worker_uuid)
    worker_number: int = 0
    connection: Optional[Connection] = None
    try:
        connection = connect_to_db()
        with connection:
            cur: Cursor = connection.cursor()
            request: str = f"SELECT worker_number FROM workers WHERE id='{worker_uuid}'"
            cur.execute(request)
            rows: List[Tuple[int]] = cur.fetchall()
            if len(rows) == 0:
                # worker not in DB
                request = "SELECT COUNT(*) as nb_worker FROM workers"
                cur.execute(request)
                row: int = cur.fetchone()[0]
                logger.debug("There is currently %s workers", row)
                worker_number = row + 1
            elif len(rows) == 1:
                worker_number = rows[0][0]
            else:
                worker_number = rows[0][0]
                # logger.error("Too many workers in DB %s", len(rows))
                # should it be an error ?
    except lite.Error as err:
        logger.error("Error in get worker number: %s", err.args[0])
    finally:
        if connection is not None:
            connection.close()

    logger.debug("Worker number is %s", worker_number)
    return worker_number


def register_job(job_uuid: str, workers: List[Dict[str, Any]], datatype: str) -> None:
    """Creates a job in DB

    Args:
        job_uuid (str): The job UUID
        workers (List[Dict[str, Any]]): The list of workers assigned to the job
        datatype (str): The datatype of the job
    """
    logger.debug("registering job %s", job_uuid)
    connection: Optional[Connection] = None
    try:
        connection = connect_to_db(development=False)
        with connection:
            cur: Cursor = connection.cursor()
            for worker in workers:
                request: str = (
                    "INSERT INTO jobs VALUES"
                    "('{}', '{}', datetime('now'), '{}', '{}')"
                ).format(job_uuid, datatype, "running", worker["id"])
                cur.execute(request)
    except lite.Error as err:
        logger.error("Error in register job: %s", err.args[0])
    finally:
        if connection is not None:
            connection.close()


def update_job_worker_state(job_uuid: str, state: str, worker_id: str) -> None:
    """Update job's state by job id and worker id

    Args:
        job_uuid (str): The job UUID
        state (str): The datatype of the job
        worker_id (str): The worker UUID
    """
    logger.debug(
        "Changing state of job %s for worker %s to %s", job_uuid, worker_id, state
    )
    connection: Optional[Connection] = None
    try:
        connection = connect_to_db(development=False)
        with connection:
            cur: Cursor = connection.cursor()
            request: str = (
                "UPDATE jobs SET state = '{}' WHERE id = '{}' AND worker_id = '{}'"
            ).format(state, job_uuid, worker_id)
            cur.execute(request)
    except lite.Error as err:
        logger.error("Error in update job worker state: %s", err.args[0])
    finally:
        if connection is not None:
            connection.close()


def update_job_state(job_uuid: str, state: str) -> None:
    """Update job's state by job id

    Args:
        job_uuid (str): The job UUID
        state (str): The datatype of the job
    """
    logger.debug("Changing state of job %s to %s", job_uuid, state)
    connection: Optional[Connection] = None
    try:
        connection = connect_to_db(development=False)
        with connection:
            cur: Cursor = connection.cursor()
            request: str = "UPDATE jobs SET state = '{}' WHERE id = '{}'".format(
                state, job_uuid
            )
            cur.execute(request)
    except lite.Error as err:
        logger.error("Error in update job state: %s", err.args[0])
    finally:
        if connection is not None:
            connection.close()


def get_local_job(job_id: str) -> Optional[Dict[str, Any]]:
    """Get job by id

    Args:
        job_id (str): Job UUID

    Returns:
        Optional[Dict[str, Any]]: The job object
    """
    logger.debug("Get job %s", job_id)
    connection: Optional[Connection] = None
    try:
        connection = connect_to_db()
        with connection:
            cur: Cursor = connection.cursor()
            cur.execute(
                (
                    "SELECT DISTINCT id, datatype, date, state FROM jobs"
                    " WHERE id='{}'"
                ).format(job_id)
            )
            rows: List[Any] = cur.fetchall()
            logger.debug(rows)
            job: Dict[str, Any] = {}
            for row in rows:
                job["id"] = row[0]
                job["datatype"] = row[1]
                job["date"] = row[2]
                job["state"] = row[3]
            job["workers"] = []
            cur.execute(
                (
                    "SELECT DISTINCT worker_id FROM jobs"
                    " WHERE id='{}' AND state != 'done'"
                ).format(job_id)
            )
            rows = cur.fetchall()
            for worker in rows:
                job["workers"].append(worker[0])

        logger.debug("Local job is: %s", job)
        return job
    except lite.Error as err:
        logger.error("Error in get local job : %s", err.args[0])
    finally:
        if connection is not None:
            connection.close()


def get_local_jobs() -> List[Dict[str, Any]]:
    """This methods returns the list of past and ongoing jobs

    Returns:
        List[Dict[str, Any]]: _description_
    """
    jobs: List[Dict[str, Any]] = []
    connection: Optional[Connection] = None
    try:
        connection = connect_to_db()
        with connection:
            cur: Cursor = connection.cursor()
            cur.execute("SELECT DISTINCT id, datatype, date, state FROM jobs")
            rows: List[Any] = cur.fetchall()
            for row in rows:
                job: Dict[str, Any] = {}
                job["id"] = row[0]
                job["datatype"] = row[1]
                job["date"] = row[2]
                job["state"] = row[3]
                jobs.append(job)
    except lite.Error as err:
        logger.error("Error in get local jobs : %s", err.args[0])
    finally:
        if connection is not None:
            connection.close()

    return jobs


def delete_jobs() -> None:
    """Delete all jobs"""
    connection: Optional[Connection] = None
    try:
        connection = connect_to_db()
        with connection:
            cur: Cursor = connection.cursor()
            cur.execute("DELETE FROM jobs")
    except lite.Error as err:
        logger.error("Error in delete jobs: %s", err.args[0])
    finally:
        if connection is not None:
            connection.close()


def delete_workers() -> None:
    """Delete all workers"""
    connection: Optional[Connection] = None
    try:
        connection = connect_to_db()
        with connection:
            cur: Cursor = connection.cursor()
            cur.execute("DELETE FROM workers")
    except lite.Error as err:
        logger.error("Error in delete workers: %s", err.args[0])
    finally:
        if connection is not None:
            connection.close()


def delete_table(table: str) -> None:
    """Delete all elements from table in database

    Args:
        table (str): The database table to empty
    """
    connection: Optional[Connection] = None
    try:
        connection = connect_to_db()
        with connection:
            cur: Cursor = connection.cursor()
            cur.execute(f"DELETE FROM {table}")
    except lite.Error as err:
        logger.error("Error in delete %s: %s", table, err.args[0])
    finally:
        if connection is not None:
            connection.close()


def get_active_workers(datatype: str = "", state: str = "") -> List[Dict[str, Any]]:
    """Get list of active workers

    Args:
        datatype (str, optional): The datatype of workers. Defaults to None.
        state (str, optional): The state of workers. Defaults to None.

    Returns:
        List[Dict[str, Any]]: The list of workers
    """
    update_state_workers()
    logger.debug("Fetching workers")
    workers: List[Dict[str, Any]] = []
    connection: Optional[Connection] = None
    try:
        connection = connect_to_db()
        with connection:
            cur: Cursor = connection.cursor()
            request: str = "SELECT * FROM workers"
            if datatype != "":
                request = f"{request} WHERE datatype == '{datatype}'"
                if state != "":
                    request = f"{request} AND state == '{state}'"
            elif state != "":
                request = f"{request} WHERE state == '{state}'"
            cur.execute(request)
            rows: List[Any] = cur.fetchall()
            for row in rows:
                worker: Dict[str, Any] = {}
                worker["id"] = row[0]
                worker["ip"] = row[1]
                worker["port"] = row[2]
                worker["datatype"] = row[3]
                worker["date"] = row[4]
                worker["state"] = row[5]
                worker["worker_id"] = row[6]
                workers.append(worker)
    except lite.Error as err:
        logger.error("Error in get workers: %s", err.args[0])
    finally:
        if connection is not None:
            connection.close()

    return workers
