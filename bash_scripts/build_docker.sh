#!/bin/bash

# Script to build the Docker image for arc-solver

set -e  # Exit on any error

IMAGE_NAME="arc-solver"
TAG="latest"

# GCP Artifact Registry variables
PROJECT_ID=$(gcloud config get-value project)
REGION=us-central1
REPO=arc-repo
IMAGE_URI=$REGION-docker.pkg.dev/$PROJECT_ID/$REPO/$IMAGE_NAME:latest

echo "Building Docker image: $IMAGE_NAME:$TAG"

# Build the Docker image
docker build -t "$IMAGE_NAME:$TAG" .

echo "Docker image built successfully: $IMAGE_NAME:$TAG"
echo "You can now run it with: docker run --rm -it $IMAGE_NAME:$TAG"
echo "Or use docker-compose: docker-compose up"

# Optional: Tag and push to GCP Artifact Registry
echo "To push to GCP Artifact Registry:"
echo "docker tag $IMAGE_NAME:$TAG $IMAGE_URI"
echo "docker push $IMAGE_URI"