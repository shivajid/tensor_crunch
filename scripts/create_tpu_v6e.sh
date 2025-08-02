#!/bin/bash
#Ensure that you have quota for TPU v6e in your Google Cloud project.
# This script creates a TPU v6e queued resource in Google Cloud.

# Make sure to set the PROJECT_ID, ACCELERATOR_TYPE, ZONE, and RUNTIME_VERSION variables before running the script.
# Please replace <your-project-id> with your actual Google Cloud Project ID.
export PROJECT_ID=<your-project-id>
export ACCELERATOR_TYPE=v6e-4
export ZONE=us-central2-b
export RUNTIME_VERSION=v2-alpha-tpuv6e
export QUEUEDRESOURCE=<your-queued-resource-name>

gcloud alpha tpu queued-resources create $QUEUEDRESOURCE \
  --accelerator-type=$ACCELERATOR_TYPE \
  --runtime-version=$RUNTIME_VERSION \
  --zone=$ZONE \
  --project-id=$PROJECT_ID 