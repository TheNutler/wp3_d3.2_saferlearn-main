#! /bin/bash
curl 'http://localhost:5000/job' -X PUT -H 'Content-Type: application/json' --data-raw '{"algorithm":"pate","datatype":"Radio","dataset":"MNIST","workers":["4", "5", "6"],"nbClasses":10,"useDP":false,"dpValue":0.05}' | jq
