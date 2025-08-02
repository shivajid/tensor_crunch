# Tensor Crunch: PEFT Tuning with JAX 
This repository provides a collection of scripts and examples for Parameter-Efficient Fine-Tuning (PEFT) of large language models, with a focus on Google's Gemma models. It includes implementations for techniques like LoRA (Low-Rank Adaptation) and FFT (Fast Fourier Transform) using both JAX/Flax and PyTorch.

The primary goal of this repository is to serve as a practical guide for fine-tuning models on custom datasets, producing efficient `safetensor` artifacts, and deploying them for inference using vLLM.

## Repository Structure

The repository is organized as follows:

-   `tensor_crunch/`: The main source directory.
    -   `trainer/`: Contains the training scripts for different model versions (Gemma-2B, Gemma-3B) and tuning methods.
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

### 1. Environment Setup

It is recommended to use a Python virtual environment.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For projects involving TPUs, you might need to follow specific setup instructions for JAX on TPU VMs.

### 2. Running Training Scripts

The training scripts are located in the `tensor_crunch/trainer/` directory. You can run a training script as follows:

```bash
python tensor_crunch/trainer/gemma3b/gemma3_1b_medical_qa_fft_trainer_vllm.py
```

Make sure to adjust the script and its parameters (e.g., dataset path, model name, output directories) as needed.

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

Once you have a verified `safetensor` checkpoint, you can deploy it for high-throughput inference using the vLLM server.

### 1. Launching the vLLM Server

Use the following command to launch the vLLM server with your fine-tuned model.

```bash
python -m vllm.entrypoints.openai.api_server \
    --model /path/to/your/gemma_safe_tensor_ckpts \
    --lora-modules /path/to/your/gemma_safe_tensor_ckpts \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.9
```

-   `--model`: Path to the base model (if the fine-tuned model is an adapter) or the full model directory.
-   `--lora-modules`: Path to the directory containing the LoRA `safetensor` adapter.
-   `--tensor-parallel-size`: Number of GPUs to use.
-   `--gpu-memory-utilization`: Fraction of GPU memory to use.

### 2. Testing the Deployed Model

You can send requests to the vLLM server using `curl` or any HTTP client.

```bash
curl http://localhost:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "/path/to/your/gemma_safe_tensor_ckpts",
        "prompt": "I have a 5 month old baby, what is the common cause of death?",
        "max_tokens": 200,
        "temperature": 0.7
    }'
```

This will send a request to the running vLLM server and return the model's completion.
