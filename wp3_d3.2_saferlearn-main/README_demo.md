# Requirements

- docker (>= 20.10.17)
- docker-compose (>= 2.6.0)

It has not been tested on inferior version of docker and docker-compose.

## Get the code

For now, use the source archive.

### Building

#### Docker images

All commands are to be executed from the root Saferlearn directory (the one containing this `README.md`).

##### Zookeeper

```bash
docker pull zookeeper
docker tag zookeeper saferlearn-zookeeper
```

##### UI

```bash
( cd ./client/ && docker build -t saferlearn-ui . )
```

##### Kafka

```bash
docker pull openjdk:8u191-jre-alpine
( cd ./kafka-docker/ && docker build -t saferlearn-kafka . )
```

##### Saferlearn

```bash
docker build -t saferlearn-platform .
```

## Usage

### Setup

- Put the public dataset in the `public-dataset` directory.
- Put the pickled private AI models in separate sub-directories in the `trained_nets` directory.
- Put the HE package (directory provided by *CEA*) under the name`pate_he/` in the `src/` directory.
- Put the FL datasets in separate sub-directories in the `fl_datasets` directory.

### Launch

You should copy the .env_template into .env and updates the .env file according to your need

```bash
./saferlearn.sh clean_data
./saferlearn.sh launch
```
To access the UI: http://<servername>:<UI_PORT> , the UI_PORT should be set in you .env file

If you are accessing the UI from a remote computer, then for the UI to be able to access the other containers you need to set up a 'ssh tunnel' from your computer
`ssh -L <ORCH_PORT>:localhost:<ORCH_PORT_EXPOSED> <servername>`

If you want to simulate a real-world use-case. You need to use data owners (or teachers) that have access to a specific dataset (and so a specific datatype). This is modelized by specifying the datatype of the teachers. To do so, in the `docker-compose_offline.yml` *teacher* service, add the following argument to the *command* field: "-t \<datatype\>".
<!-- FIXME : This requires a change in the UI to accept incoming datatype -->

To access the UI: http://<servername>:<UI_PORT> , the UI_PORT should be set in you .env file

If you are accessing the UI from a remote computer, then for the UI to be able to access the other containers you need to set up a 'ssh tunnel' from your computer
`ssh -L <ORCH_PORT>:localhost:<ORCH_PORT_EXPOSED> <servername>`

### Running

In order to launch the demo, run the command

```bash
./saferlearn.sh launch <optional_number_of_workers>
```

On the UI, select :

- the datatype
- the number of teachers
- the aggregation method (FHE, MPC or *FL*).

If when chaining multiple runs the database become corrupted and the UI shows too many nodes compared to the number of services launched, flush the database with the UI before launching a new job.

#### FHE

To retrieve the aggregated votes you need to run the command

```bash
./saferlearn.sh get-results-pate-fhe
```

Then you can check into the `./results/` directory to get the final labels. Otherwise the results are in the following folder : `saferlearn-framework/src/pate-he/student/data/result`.

#### MPC

You may need to create certificates for MPC. To do so, enter a running saferlearn-platform container and create the certificates:

```bash
docker exec -it saferlearn_integration_teacher_1 bash
# and then inside the container :
./Scripts/setup-ssl.sh <number of computing parties>
./Scripts/setup-clients.sh <number of teachers>
```

The results are in `src/result.file`.

#### FL

Launch the FL set-up using the bash script:

```bash
./saferlearn.sh launch <optional_number_of_workers>
```

It is based on the docker-compose_offline.yml to launch the corresponding containers.
Do not forget to build the correct Docker Image (cf Building - Saferlearn section)
To stop these containers, use:

```bash
./saferlearn.sh down
```

You need to generate a ssh key pair (public key `id_rsa.pub` and private key `id_rsa` under `tff/credentials/ssh`). You can also add the repository `tff/credentials/ssh/authorized_keys` and copy `id_rsa.pub` in it. Make sure to have the correct permissions for these repository/files.

For now, the dataset are stored on `fl_datasets` as csv files, under a repository `_index` (i.e. 0, 1, 2, 3, ....).
Note: The filepaths of the participants are to be defined in a specific file similar to src/usecases/UCStub.py, the FlUtils class should then be imported in the src/tff_central.py

```bash
mkdir -p src/input-data/<dataset_name>
```

For now, it is only working with dataset_name == cifar10_train (up to 10 datasets in the repository)

### Debugging

If you want to get the logs of the different containers while running Saferlearn, run :

```bash
./saferlearn.sh logs
```

To clean the output from a run, you can use the `clean_data.sh` script. You may need to run it as root since files created by Docker are usually owned by root.

## Change Model

You need to store your models in sub directories (1, 2, 3, .....) located in the directory `trained_nets`

## Change Dataset

You need to update the function `get_dataset` in the file `data_owner.py` to pre process and import
correctly the dataset that you want to use as a public dataset.

## API

To check the number of worker that are registered call the API with a `GET` on `/clients`.

To launch a job, call the API with a `PUT` on `/job`.
