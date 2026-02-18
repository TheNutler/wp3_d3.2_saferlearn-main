# Saferlearn

Saferlearn is a Secure collaboration framework allowing participant to train their models securely in a collaborative manner. It preserve privacy of the underlying data and avoid any private data sharing.

It relies on techniques such as Private Aggregation of Teacher Ensemble (PATE), Multi Party Computation (MPC) or Federated learning and technologies such as Homomorphic encryption and differential privacy.

## Get the code

```bash
git clone https://gitlab.thalesdigital.io/friendlyhackers/secure-collaborative-learning/saferlearn/saferlearn-framework.git
````

### Building

#### Docker images

All commands are to be executed from the root Saferlearn directory (the one containing this `README.md`).

Saferlearn relies on Kafka to do all the communications between participants.

```

#### Data Owner

##### Requirements

```bash
pip install -r ./requirements.txt
```

Copy you trained model inside the directory `trained_nets_gpu/0`

##### Usage

In order to launch a data owner you need to run the data_owner_cli.py script

```bash
 Usage: data_owner_cli.py [OPTIONS] [PROGRAM]                                                                                                                                                                  
                                                                                                                                                                                                               
╭─ Arguments ─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│   program      [PROGRAM]  Client program to run by the teachers [default: pate-teacher]                                                                                                                     │
╰─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
╭─ Options ───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ --type                            TEXT     Node type, default 'Stub' [default: Stub]                                                                                                                        │
│ --computing-parties-number        INTEGER  Number of computing parties [default: 3]                                                                                                                         │
│ --nteachers                       INTEGER  Number of teachers [default: 3]                                                                                                                                  │
│ --size-batch                      INTEGER  Size of batch [default: 10]                                                                                                                                      │
│ --sigma                           INTEGER  Sigma for Gaussian noise distribution [default: 1]                                                                                                               │
│ --rounds                          INTEGER  Number of rounds for noise [default: 1]                                                                                                                          │
│ --data-format                     TEXT     Your type of data [default: radio]                                                                                                                               │
│ --rpyc-host                       TEXT     Your listening address [default: 127.0.0.1]                                                                                                                      │
│ --rpyc-port                       INTEGER  Your listening port [default: 1244]                                                                                                                              │
│ --orchestrator-host               TEXT     Orchestrator ip address or hostname [default: 127.0.0.1]                                                                                                         │
│ --orchestrator-port               INTEGER  Orchestrator listening port [default: 5000]                                                                                                                      │
│ --help                                     Show this message and exit.                                                                                                                                      │
╰─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯   
```

#### Change Model

You need to store your models in sub directories (0, 1, 2, 3, .....) located in the directory `trained_nets_gpu`.

#### Adding a new participant

In order to add a new participant (i.e. data owner) one needs to implement a class extending `usecases.data_owner_abstract_class.py`. An example is available in `usecases.data_owner_example.py`

Note: The filepaths of the participants' dataset are to be defined in a specific file similar to `src/usecases/UC_stub.py`

```bash
mkdir -p src/input-data/<dataset_name>
```

## Change Dataset

You need to update the function `get_dataset` in the file `data_owner.py` to pre process and import
correctly the dataset that you want to use as a public dataset.

#### Orchestrator

The orchestrator is the component that handle the orchestration and creation of the several jobs. It put in contacts participants with the same data type to generate a collaborativly learnt model.

##### Requirements

```bash
pip install -r ./requirements.txt
```

##### Usage

In order to launch the orchestrator and the api you need to run the api.py script

```bash
 Usage: api.py [OPTIONS]                                                                                                                                                                                       
                                                                                                                                                                                                               
╭─ Options ───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ *  --datasets-path        TEXT  Public datasets path [default: None] [required]                                                                                                                             │
│    --help                       Show this message and exit.                                                                                                                                                 │
╰─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
```

#### Model Owner

TODO

#### MPC

You may need to create certificates for MPC. To do so, enter a running saferlearn-platform container and create the certificates:

```bash
docker exec -it saferlearn_integration_teacher_1 bash
# and then inside the container :
./Scripts/setup-ssl.sh <number of computing parties>
./Scripts/setup-clients.sh <number of teachers>
```

The results are in `src/result.file`.

## API

To check the number of worker that are registered call the API with a `GET` on `/clients`.

To launch a job, call the API with a `PUT` on `/job`.

# TL;DR

## Launch

Before launching the framework you need to create and update the `.env` file.

```bash
cp .env_template .env
````

### Docker Compose

```bash
git clone https://gitlab.thalesdigital.io/friendlyhackers/secure-collaborative-learning/saferlearn/saferlearn-framework.git
cd saferlearn-framework
source .env; docker compose -p $NAME -f ./docker-compose_offline.yml --env-file ./.env up -d --scale teacher=4
```

### CLI

```bash
git clone https://gitlab.thalesdigital.io/friendlyhackers/secure-collaborative-learning/saferlearn/saferlearn-framework.git
cd saferlearn-framework
docker compose -f ./kafka_docker.yml up -d # You might have to tweak the subnet CIDR
python -m venv .venv && source .venv/bin/activate
pip install -r ./requirements.txt
python3 ./src/api.py  --datasets-path ./src/input-data/ &
python3 ./saferlearn-framework/src/data_owner_cli.py --data-format radio --rpyc-port 1234 &
```

## List workers

```bash
curl -X PUT -H 'Content-Type: application/json' -i http://localhost:5000/clients
```

## Create a job

```bash
curl -X PUT -H 'Content-Type: application/json' -i http://localhost:5000/job --data '{
  "algorithm": "pate",
  "datatype": "radio",
  "dataset": "cifar10train",
  "workers": "2",
  "nbClasses": "10",
  "dpValue": "0.5",
  "useDP": "false"
}'
```

Available algorithms:

* pate
* fhe
* fl
* mpc
