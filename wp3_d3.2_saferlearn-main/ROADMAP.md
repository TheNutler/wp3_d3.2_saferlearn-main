# Roadmap

## Architecture

```bash
~> Saferlearn
|-- Saferlearn framework
    |-- MPC PATE
    |   |-- MP-SPDZ
    |-- FL ?
    |-- Watermarking ?
    |-- Unlearning ?
    |-- HE
|-- client
|   |-- front
|   |-- server
```

## Notes

- ~~what does the result mean ? (0/2 or 0/1)~~
- why do we need datatype ? how can we find them ?
  - is it fine to allow any dataset disregarding the datatype ?
- ~~what is the model uri ? ('100')~~
- why share /var/ with containers
- public-dataset : The dataset (the training partition will be used by teachers to make predictions)
- trained_nets : The private models (100 available)

## Todo

- ~~add safeguards in scripts (especially clean_data) to avoid system destruction if all variables are not set~~
- ~~handle keyGen with Python instead of script (specify correctly the number of teachers)~~
  - add keyGen / initialization button
- ~~change alpha in LWE parameters~~
- ~~changes strings path to Python Path~~
- ~~handle clean data, create directories through python~~
- update images
  - ~~kafka~~
  - ~~zookeeper~~
  - saferlearn
    - update MP-SPDZ
    - use MP-SPDZ image to build executables and then copy to Saferlearn container
  - ~~ui~~
- standardize the notation : teachers/students vs model_owner/api
- ~~remove some kafka logs~~
- ~~remove method specification in model_owner.py call (use value from ui)~~
- ~~deprecate setup.sh~~
- verify all functions returns (especially subprocess run return code)
- ~~send keys through kafka instead of cp~~
- specify student type in kafka with ui instead of in docker-compose ?
- ~~update the Readme~~
- when using MPC, the student should get the final result instead of the teachers probably
- return results of prediction (at least for mpc) in stdout to avoid result.file
  - put result.file of mpc somewhere easy to find
- check the data provenance when using MPC
- distribute the dataset with Kafka or public link
- ~~get rid of hardcoded HOSTS files~~
- ~~Rename docker image to saferlearn-platform from mpc-fhe-saferlearn~~
- ~~improve commands in readme.md with docker context (docker exec ...)~~
- handle case of teacher disconnection in PATE
- Make kafka endpoint into a configuration file
- verify we correctly add noise in every scenarios
- ~~docker-compose down for replicas~~
- ~~speedup UI flush~~
- update UI to accept datatype dynamically
  - pipe a getJobs/getWorkers after sending the delete
- change input-data folder into public-dataset_teachers_predictions
- get rid of src/Pipfile, seems useless
- ~~do dataset separation only once~~
  - ~~do multiple folders~~
  - ~~this ensures every teachers and the student use the same data~~
- ~~Fix he aggregation without DP output~~
- pate +he + dp : results can be 0,1,2,3,4 for 4 classes: there is an issue
