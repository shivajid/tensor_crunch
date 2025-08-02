# Tensor Crunch: LLM Fine-Tuning with JAX/Tunix and Maxtext

This repository provides a collection of scripts and examples for Parameter-Efficient Fine-Tuning (PEFT) of large language models, with a focus on Google's Gemma models. It includes implementations for techniques like LoRA (Low-Rank Adaptation) and SFT using both JAX/Flax. These tutorials are build using Google's [Tunix](https://github.com/google/tunix.git). You can read more about it in the  ([docs](https://deepwiki.com/google/tunix/1-overview)).

The primary goal of this repository is to serve as a practical guide for fine-tuning models on custom datasets, producing efficient `safetensor` artifacts, and deploying them for inference using vLLM.

## Repository Structure

The repository is organized as follows:

-   `tensor_crunch/`: The main source directory.
    -   `trainer/`: Contains the training scripts for different model versions (Gemma-2B, Gemma-3B) and tuning methods. **NOTE** - Some of the files may be WIP. Please use the main trainer in the root of the file.
        -   `gemma2b/`: Scripts for Gemma-2B models.
        -   `gemma3b/`: Scripts for Gemma-3B models.
    -   `utility/`: Contains utility scripts, including a script to test `safetensor` checkpoints.
    -   `scripts/`: Helper scripts for environment setup.
    -   `docs/`: Additional documentation and guides.
    -   `data.py`: Data loading and preprocessing scripts.
    -   `Dockerfile`: Dockerfile for containerized environment.
    -   `requirements.txt`: Python dependencies.
    -   `README.md`: This file.

## Getting Started

### Hardware Requirements
The ([tunix](https://github.com/google/tunix.git)) library is built on jax and  targetted for TPUs. In the example we assume that you have TPU v6e-4 machine to work through the examples. 
You can refer to the ([script](scripts/create_tpu_v6e.sh) to see how to create  TPU v6e VM. 

The tutorial is for single hots TPUs (V6e-1,4 or 8) worker configurations. 

### 1. Environment Setup

It is recommended to use a Python virtual environment. Tunix required python 3.11 for all the dependencies to work. The ([script](scripts/create_python_env.sh)) has scripts on how to setup python3.11 on your TPUV

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
It is possible that some packages may have missed. Install it as you see fit.

### KAGGLE, HF and WANDB

The trainer uses 
   -   Kaggle to download the weights and 
   -   HuggingHF for the datasets
   -   WANDB for training monitoring

Ensure that you have an account and security keys setup.

**Kaggle**

```bash
        export KAGGLE_USERNAME=user_name
        export KAGGLEHUB_CACHE=<path> # default writes to .cache
        export KAGGLE_KEY=<key>
```
**WANDB**

```bash
    export WANDB_API_KEY=<key>
```
**HUGGINGFACE**
```bash
    export HUGGINGFACE_TOKEN=<key>
```



### 2. Running Training Scripts

The main training script is `tensor_crunch/gemma3_1b_fft_trainer.py`. This script handles the entire training pipeline, including data loading, model configuration, training, and saving checkpoints.

To run the training script, use the following command structure. You must provide the required directory paths for checkpoints, profiling, and servable models.

**Example:**

```bash
python tensor_crunch/gemma3_1b_fft_trainer.py \
    --intermediate_ckpt_dir ./output/intermediate_ckpt/fft/v3/ \
    --ckpt_dir ./output/ckpts/medical/fft/v3/ \
    --profiling_dir ./output/profiling/ \
    --servable_ckpt_dir ./output/servable_ckpt/fft/v1/ \
    --batch_size 16 \
    --rank 16 \
    --max_steps 500 \
    --dataset_name "medalpaca/medical_meadow_medqa"
```

You can customize the training by modifying the arguments. For a full list of available arguments and their descriptions, refer to the docstring within the `tensor_crunch/gemma3_1b_fft_trainer.py` script. For more details, refer to the [Gemma 3B Trainer Guide](tensor_crunch/docs/gemma3_1b_fft_trainer_guide.md).

## Verifying Safetensors

After training, a `safetensor` file is created. You can verify this checkpoint using the `tensor_crunch/utility/hf_safetensor_test.py` script.

The script performs the following steps:

1.  **Load Tokenizer and Model**: It loads the tokenizer and the fine-tuned model from the specified local directory where the `safetensors` are stored.
2.  **Prepare Prompt**: A sample prompt is prepared to test the model's response.
3.  **Run Inference**: The model generates a response based on the prompt.
4.  **Decode and Print**: The generated output is decoded and printed to the console.

Here is how to use it:

1.  Open `tensor_crunch/utility/hf_safetensor_test.py`.
2.  Modify the `local_model_path` variable to point to the directory containing your `model.safetensors` file and tokenizer configuration.

    ```python
    local_model_path = "/path/to/your/gemma_safe_tensor_ckpts"
    ```

3.  Run the script:

    ```bash
    python tensor_crunch/utility/hf_safetensor_test.py
    ```

4.  Check the output to see the model's response.

## Deploying with vLLM

Once you have a verified `safetensor` checkpoint, you can deploy it for high-throughput inference using the vLLM server. This can work both on GPUs and TPUs. The script has been tested with vLLM on TPUs.

For vLLM on TPU you can use the following ([instructions](https://cloud.google.com/tpu/docs/tutorials/LLM/vllm-inference-v6e)) to build vLLM on TPU to serve your model.

The safetensor checkpoints and the files are generated in loca path and you may need to move to GCS for copying across your training serving infrasturcture.

### 1. Launching the vLLM Server

The vLLM server most likely will be on a different machine and accelerator type than the the training node.

Once you have setup vLLM on the target ([node](https://docs.vllm.ai/en/latest/getting_started/installation/index.html)) you will need to start the vLLM pointing the mode to the safetensor and giving a name. Below is a sample script.

```bash
python -m vllm.entrypoints.openai.api_server \
    --model /path/to/your/gemma_safe_tensor_ckpts \
    --tensor-parallel-size 4 \
    --served-model-name <model_name> \
    --host 0.0.0.0 --port 8000 --download_dir /tmp \
    --swap-space 16 --disable-log-requests \
    --gpu-memory-utilization 0.9
```
E.g.

```bash
python -m vllm.entrypoints.openai.api_server \ 
        --model /home/shivajid/ckpts/gemma3_1b_medical_qa_fft/ \ 
        --served-model-name "medqa" \ 
        --host 0.0.0.0 --port 8000 --download_dir /tmp \
        --swap-space 16 --disable-log-requests \
        --tensor_parallel_size=4 \
        --max-model-len=204
```
-   `--model`: Path to the base model (if the fine-tuned model is an adapter) or the full model directory.
-   `--served-model-name`: Name of the served model. This will be used when calling from the client.
-   `--tensor-parallel-size`: Number of GPU or TPU chips to use.

### 2. Testing the Deployed Model

You can send requests to the vLLM server using `curl` or any HTTP client.

```bash
curl http://localhost:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": <served_model_name>,
        "prompt": "Prompt Text",
        "max_tokens": 200,
        "temperature": 0.7
    }'
```

This will send a request to the running vLLM server and return the model's completion.

## Agent Devleopment Kit

You can interact with the model if you work with Agent Development kit. Currently ADK only supports VertexAI deployment for custom models. You can deploy the Gemma models on a GPU(vLLM) or TPU(Hex-LLM). 