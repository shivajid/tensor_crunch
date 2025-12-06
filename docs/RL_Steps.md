## Maxtext RL Training
## Login
### Ensure you have run gcert
```
gcloud compute ssh --zone $zone $vm_name --project $project_id
```
### Activate pip env

```
source maxtext_env/bin/activate
```
### Build
### Code Changes
### The Tunix, vllm and tpu-inference are in parallel folder to maxtext, you can update the code there
### The build command generates a local docker image with the name

```
cd maxtext
bash dependencies/scripts/docker_build_dependency_image.sh MODE=post-training-experimental POST_TRAINING_SOURCE=local
```

### Push docker to AR

### List docker images
```
docker images
```
### Give a Image Name
```
IMAGE_NAME=mvt-build-1204-run07
export DOCKER_IMAGE="us-east5-docker.pkg.dev/$PROJECT_ID/rl-maxtext/${IMAGE_NAME}:latest
"
```
### Tag Docker image
```
docker tag maxtext_base_image:latest $DOCKER_IMAGE
```
### push the docker
```
docker push $DOCKER_IMAGE
```
### Set a Workload Id, e.g.
```
export WORKLOAD_ID=mvt-build-1204-run07
```
### Now you are ready to submit your job

### This is for the 70b job on v5p-64(32 chip)

```
xpk workload create-pathways  --workload $WORKLOAD_ID  --docker-image $DOCKER_IMAGE  --cluster $CLUSTER_ID  --tpu-type=v5p-64  --num-slices=1  --zone=$ZONE  --project=$PROJECT_ID  --priority=high  --command "HF_TOKEN=$hf_token TF_CPP_MIN_LOG_LEVEL=0 JAX_PLATFORMS=proxy JAX_BACKEND_TARGET=grpc://127.0.0.1:29000 ENABLE_PATHWAYS_PERSISTENCE='1' python3 -m src.MaxText.rl.train_rl src/MaxText/configs/rl.yml model_name=llama3.1-70b tokenizer_path=meta-llama/Llama-3.1-70B-Instruct load_parameters_path=gs://runner-maxtext-logs/2025-11-24-02-54/llama3.1-70b/scanned_chkpt/0/items run_name=$WORKLOAD_ID base_output_directory=gs://maxtext-shivajid/rl/output hf_access_token=$hf_token checkpoint_storage_use_ocdbt=False checkpoint_storage_use_zarr3=False profiler=xplane batch_size=16"
```
### This is for the 8b job on v5p-32(32 chip)

```
xpk workload create-pathways  --workload $WORKLOAD_ID  --docker-image $DOCKER_IMAGE  --cluster$CLUSTER_ID  --tpu-type=v5p-32  --num-slices=1  --zone=$ZONE  --project=$PROJECT_ID  --priority=high  --command
"HF_TOKEN=$hf_token TF_CPP_MIN_LOG_LEVEL=0 JAX_PLATFORMS=proxy JAX_BACKEND_TARGET=grpc://127.0.0.1:29000 ENABLE_PATHWAYS_PERSISTENCE='1' python3 -m src.MaxText.rl.train_rl src/MaxText/configs/rl.yml model_name=llama3.1-8b tokenizer_path=meta-llama/Llama-3.1-8B-Instruct load_parameters_path=gs://runner-maxtext-logs/2025-12-05-23-35/llama3.1-8b/scanned_chkpt/0/items run_name=$WORKLOAD_ID base_output_directory=gs://maxtext-shivajid/rl/output hf_access_token=$hf_token checkpoint_storage_use_ocdbt=False checkpoint_storage_use_zarr3=False profiler=xplane batch_size=4"
```
### Check the pods
```
kubectl get pods
```
### Check the logs
```
kubectl logs -f <podname>
```
### Check details on the image
```
kubectl describe pod <podname>
```
