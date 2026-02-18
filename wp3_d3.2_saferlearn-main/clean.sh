#!/bin/bash

./clean_data.sh
./clean_results.sh

rm -rf src/pate_he/student/data/ciphertexts/*
rm -rf src/pate_he/aggregator/data/ciphertexts/encrypted_teacher_votes/*
rm -rf src/pate_he/teacher/data/ciphertexts/*
rm -rf src/pate_he/teacher/data/teacher_votes/*

rm src/result.file

rm trained_nets/*/.lock
