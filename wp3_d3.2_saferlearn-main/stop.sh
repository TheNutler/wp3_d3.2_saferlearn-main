#!/bin/bash

docker container stop `docker ps -q`
docker container rm `docker container ls -a -q`
docker volume rm `docker volume ls -q`
