#!/bin/bash
export SAFERLEARN_VERSION=1.2.0

ENV_FILE=.env
ENGINE=docker
COMPOSE_ENGINE=docker-compose
COMPOSE_FILE=docker-compose_offline.yml

HE_PATE_DIR="./src/pate_he"
MODELS_DIR="./trained_nets/"

RED="\e[31m"
BLUE="\e[34m"
GREEN="\e[32m"
ENDCOLOR="\e[0m"

echo -e "${BLUE}Saferlearn command line${ENDCOLOR}"

function loading_env {
    if [[ ! -f ${ENV_FILE} ]]; then
        echo -e "${RED} env file ${ENV_FILE}${ENDCOLOR} is missing you should create it using the .env_template, have look to the README.md"
        exit 1
    fi

    echo -e "${GREEN}Getting env variable from ${ENV_FILE}${ENDCOLOR}"
    unamestr=$(uname)
    if [ "$unamestr" = 'Linux' ]; then

        export $(grep -v '^#' ${ENV_FILE} | xargs -d '\n')

    elif [ "$unamestr" = 'FreeBSD' ] || [ "$unamestr" = 'Darwin' ]; then

        export $(grep -v '^#' ${ENV_FILE} | xargs -0)

    fi
}

function launch {
    NB_WORKERS=$1
    [ -z "$NB_WORKERS" ] && NB_WORKERS=3
    loading_env
    echo -e "${BLUE}Starting Saferlearn network with $NB_WORKERS worker(s)${ENDCOLOR}"
    echo "${COMPOSE_ENGINE} -p ${NAME} -f ${COMPOSE_FILE} up -d --scale teacher=$NB_WORKERS --scale aggregator=3"
    ${COMPOSE_ENGINE} -p "${NAME}" -f ${COMPOSE_FILE} up -d --scale teacher=$NB_WORKERS --scale aggregator=3
}

function down {
    loading_env
    echo -e "${BLUE}Setting down Saferlearn network${ENDCOLOR}"
    echo "${COMPOSE_ENGINE} -p ${NAME} -f ${COMPOSE_FILE} down -v"
    ${COMPOSE_ENGINE} -p "${NAME}" -f ${COMPOSE_FILE} down -v
}

function status {
    loading_env
    ${COMPOSE_ENGINE} -p "${NAME}" -f ${COMPOSE_FILE} ps -a
}

function logs {
    loading_env
    local CONTAINER=$1
    ${COMPOSE_ENGINE} -p "${NAME}" -f ${COMPOSE_FILE} logs -f $CONTAINER
}

function clean-docker {
    echo -e "${BLUE}Cleanning all docker container and volumes${ENDCOLOR}"
    echo -e "${RED}CAUTION !!!! this will affect other users as well${ENDCOLOR}"
    read -rp "Continue (y/n)?" choice
    case "$choice" in
    y | Y)
        echo "yes"
        ${ENGINE} container stop $(docker ps -q)
        ${ENGINE} container rm $(docker container ls -a -q)
        ${ENGINE} volume rm $(docker volume ls -q)
        ;;
    n | N) echo "no" ;;
    *) echo "invalid" ;;
    esac

}

function clean-data {
    loading_env
    echo "Cleaning data files"
    find ${HE_PATE_DIR}/*/data/*/* -delete
    echo "Cleaning database"
    rm ./src/worker.db

    echo "Create directories if required"

    mkdir -p "${HE_PATE_DIR}/aggregator/data/ciphertexts/encrypted_argmax_folder"
    mkdir -p "${HE_PATE_DIR}/aggregator/data/ciphertexts/encrypted_teacher_votes"
    mkdir -p "${HE_PATE_DIR}/aggregator/data/keys"

    mkdir -p "${HE_PATE_DIR}/student/data/ciphertexts"
    mkdir -p "${HE_PATE_DIR}/student/data/result"
    mkdir -p "${HE_PATE_DIR}/student/data/keys"

    mkdir -p "${HE_PATE_DIR}/teacher/data/ciphertexts"
    mkdir -p "${HE_PATE_DIR}/teacher/data/keys"
    mkdir -p "${HE_PATE_DIR}/teacher/data/teacher_votes"

    echo "Set executable bit for executables"

    find ${HE_PATE_DIR}/*/bin/* -type f -exec chmod 755 {} \;

    echo "Removing .lock in the models directory ${MODELS_DIR}"

    find ${MODELS_DIR}/. -name ".lock" -delete
}

function get-results-pate-fhe {
    echo -e "${BLUE}Requesting the results from the Saferlearn framework${ENDCOLOR}"
    mkdir -p ./results/
    mv ./src/pate_he/student/data/result/* ./results/
    sed -i 's/2/1/g' ./results/*
    echo "The decrypted labels are available in the results directory"
}

function help {
    # Display Help
    loading_env
    echo
    echo -e "Syntax: saferlearn.sh ACTION ACTION_ARGUMENTS"
    echo -e "${GREEN}ACTION possibilities${ENDCOLOR}"
    echo -e "${BLUE}launch${ENDCOLOR} : \tone argument, the number of worker. \n\t\t${GREEN}Launch the compose file ${COMPOSE_FILE} on workspace ${NAME}${ENDCOLOR}"
    echo -e "${BLUE}down${ENDCOLOR} : \t\tno argument. \n\t\t${GREEN}Set down the compose file ${COMPOSE_FILE} on workspace ${NAME}.${ENDCOLOR}"
    echo -e "${BLUE}status${ENDCOLOR} : \tno argument. \n\t\t${GREEN}Get the status the compose file ${COMPOSE_FILE} on workspace ${NAME}.${ENDCOLOR}"
    echo -e "${BLUE}logs${ENDCOLOR} : \t\tseveral aguments, one per container or service (you should use the service name of the compose file). \n\t\t${GREEN}Get the logs of the selected services from the compose file ${COMPOSE_FILE} on workspace ${NAME}${ENDCOLOR}"
    echo -e "${BLUE}clean-data${ENDCOLOR} : \tno arguments. \n\t\t${GREEN}Cleaning the data for saferlearn framework.${ENDCOLOR}"
    echo -e "${BLUE}get-results-pate-fhe${ENDCOLOR} : \tno arguments. \n\t\t${GREEN}Obtaining the PATE HE results in the ./results folder.${ENDCOLOR}"
    echo
}

ACTION=$1
case $ACTION in

launch)
    launch "${@:2}"
    ;;

down)
    down
    ;;

status)
    status
    ;;

logs)
    logs "${@:2}"
    ;;

clean-data)
    clean-data
    ;;

get-results-pate-fhe)
    get-results-pate-fhe
    ;;

help)
    help
    ;;

*)
    help
    ;;
esac
