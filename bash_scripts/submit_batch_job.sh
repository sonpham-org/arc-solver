#!/bin/bash

# Script to submit a Google Cloud Batch job to run the arc-solver software

set -e  # Exit on any error

# Google Cloud Batch parameters
JOB_NAME="arc-solver-batch-$(date +%Y%m%d-%H%M%S)"
LOCATION="us-central1"  # Adjust location as needed
PROJECT_ID=$(gcloud config get-value project)
REGION="us-central1"
REPO="arc-repo"
IMAGE_NAME="arc-solver"
IMAGE_URI="$REGION-docker.pkg.dev/$PROJECT_ID/$REPO/$IMAGE_NAME:latest"

# Job configuration
MACHINE_TYPE="n1-standard-4"  # Adjust machine type as needed
TASK_COUNT=1
MAX_RUN_DURATION="3600s"  # 1 hour, adjust as needed

echo "Submitting Google Cloud Batch job: $JOB_NAME"
echo "Using image: $IMAGE_URI"

# Submit the batch job
gcloud batch jobs submit $JOB_NAME \
  --location=$LOCATION \
  --config=- <<EOF
{
  "taskGroups": [
    {
      "taskSpec": {
        "runnables": [
          {
            "container": {
              "imageUri": "$IMAGE_URI",
              "commands": [
                "python",
                "src/main.py",
                "--challenges",
                "data/arc-2024/arc-agi_training_challenges.json",
                "--solutions",
                "data/arc-2024/arc-agi_training_solutions.json",
                "--limit",
                "10",
                "--clear-responses",
                "--num-initial-generations",
                "10",
                "--max-reflections",
                "3"
              ],
              "volumes": [
                {
                  "gcs": {
                    "remotePath": "gs://your-bucket/output"  # Replace with your GCS bucket
                  },
                  "mountPath": "/app/output"
                },
                {
                  "gcs": {
                    "remotePath": "gs://your-bucket/db"  # Replace with your GCS bucket
                  },
                  "mountPath": "/app/db"
                },
                {
                  "gcs": {
                    "remotePath": "gs://your-bucket/data"  # Replace with your GCS bucket
                  },
                  "mountPath": "/app/data"
                }
              ],
              "environment": {
                "GEMINI_API_KEY": "\${GEMINI_API_KEY}"  # Assumes secret is set
              }
            }
          }
        ],
        "computeResource": {
          "cpuMilli": 4000,
          "memoryMib": 8192
        },
        "maxRunDuration": "$MAX_RUN_DURATION"
      },
      "taskCount": $TASK_COUNT,
      "parallelism": 1
    }
  ],
  "allocationPolicy": {
    "instances": [
      {
        "policy": {
          "machineType": "$MACHINE_TYPE"
        }
      }
    ]
  },
  "logsPolicy": {
    "destination": "CLOUD_LOGGING"
  }
}
EOF

echo "Google Cloud Batch job submitted successfully: $JOB_NAME"
echo "You can check the job status with: gcloud batch jobs describe $JOB_NAME --location=$LOCATION"