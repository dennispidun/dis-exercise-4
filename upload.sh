#!/bin/bash

for i in $(seq 1 1); do 
  echo step1
  gcloud compute scp --recurse ../dis-execise-4 instance-$i:dis-exercise-4
  echo step2
	cat <<EOT > docker-compose.yml
version: '3'
services:
  dis4:
    container_name: dis-uebung-4
    environment:
      TF_CONFIG: '{"cluster": {"worker": ["10.186.0.6:2222", "10.186.0.7:2222", "10.186.0.8:2222", "10.186.0.9:2222"]}, "task": {"type": "worker", "index": $((i-1))}}'
    build: .
    ports:
    - '2222:2222'
EOT
  echo step3
	gcloud compute scp docker-compose.yml instance-$i:dis-exercise-4/docker-compose.yml
  echo step4
	nohup gcloud compute ssh instance-$i --command 'cd dis-exercise-4 && docker-compose down && docker-compose up -d --build' &
done