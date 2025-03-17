#!/bin/bash
# push_image.sh
# This script builds and pushes the Docker image to Amazon ECR.
ECR_REPO_URI=<YOUR_ECR_REPO_URI>
IMAGE_TAG=latest
docker build -t llm-app .
docker tag llm-app:latest ${ECR_REPO_URI}:${IMAGE_TAG}
docker push ${ECR_REPO_URI}:${IMAGE_TAG}
