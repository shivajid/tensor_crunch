# PEFT Tuning of Gemma Models with JAX and PyTorch

This repository provides a collection of scripts and tools for fine-tuning Google's Gemma models using Parameter-Efficient Fine-Tuning (PEFT) techniques such as LoRA (Low-Rank Adaptation) and  Full Fine tuning (FFT). The primary frameworks used are JAX with Tunix. The traning checkpoints are saved in Orbax and then converted to `safetensors` format to be able to served by vLLM server.

## Project Overview

The main goal of this project is to offer a practical, hands-on environment for experimenting with different fine-tuning strategies on Gemma models. Key features include:

*   **PEFT Techniques:** Implementations for LoRA and FFT to adapt large models with minimal computational overhead.
*   **Multiple Frameworks:** Support for JAX , allowing for flexibility in training environments.
*   **Models:** Focused on the Gemma family of models (e.g., Gemma 2B and 3B). This can be extended to supported models of Tunix.
*   **Secure Serialization:** Outputs trained models in the `safetensors` format for enhanced security and performance.
*   **Deployment Ready:** Includes examples and guidance for deploying fine-tuned models with vLLM for high-performance inference.

## Repository Structure

The repository is organized into the following key directories:

*   `tensor_crunch/trainer/`: Contains the core training scripts, categorized by Gemma model versions (e.g., `gemma2b`, `gemma3b`). Each subdirectory includes scripts for different fine-tuning methods and datasets.
*   `tensor_crunch/utility/`: Includes helper scripts and utilities, such as the `hf_safetensor_test.py` script for verifying trained models.
*   `tensor_crunch/scripts/`: Contains shell scripts for setting up the environment, like `create_python_env.sh`.
*   `tensor_crunch/docs/`: Provides additional documentation and guides for specific trainers.

## Getting Started

To begin, set up your environment and run a training script.

### Hardware Requirements

The training scripts in this repository are designed to run on Google Cloud TPUs. Specifically, they are optimized for **TPU v6e** instances. The default configuration is set up to run on a `v6e-4` machine.

### 1. Environment Setup

Create a Python virtual environment and install the required dependencies.

```bash
# Create and activate a Python 3.11 virtual environment
./tensor_crunch/scripts/create_python_env.sh

# Activate the environment
source .venv/bin/activate

# Install dependencies
pip install -r tensor_crunch/requirements.txt
```

### 2. Run a Training Script

To run the main training script for Gemma 3.1B with FFT, use the following command. This example includes all the required arguments to get started.

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

For more detailed information on the trainer and its parameters, please refer to the [Gemma 3.1B FFT Trainer Guide](./docs/gemma3_1b_fft_trainer_guide.md).

## Verifying Safetensors

After training, your model is saved in the `safetensors` format. You can verify the model by running the provided test script `tensor_crunch/utility/hf_safetensor_test.py`. This script loads a fine-tuned model and runs inference to ensure it's working correctly.

### How to Use

1.  **Update the Model Path**: Open [`tensor_crunch/utility/hf_safetensor_test.py`](tensor_crunch/utility/hf_safetensor_test.py:12) and update the `local_model_path` variable to point to your checkpoint directory.

    ```python
    # Define the path to your local model directory
    local_model_path = "/path/to/your/gemma_safe_tensor_ckpts"
    ```

2.  **Run the script**:
    ```bash
    python tensor_crunch/utility/hf_safetensor_test.py
    ```

The output will show the model's response to the prompt in the script: `"I have a 5 month old baby, what is the common cause of death? "`.

## Deploying with vLLM

You can deploy your `safetensors` model for high-throughput inference using a vLLM server.

### 1. Launch the vLLM Server

Start the vLLM server and point it to your local model directory.

```bash
python -m vllm.entrypoints.openai.api_server \
    --model /path/to/your/gemma_safe_tensor_ckpts \
    --served-model-name gemma-lora-tuned
```

### 2. Test the Deployed Model

You can now send requests to the vLLM server using a tool like `curl` or any HTTP client.

```bash
curl http://localhost:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "gemma-lora-tuned",
        "prompt": "What is the capital of France?",
        "max_tokens": 50,
        "temperature": 0.7
    }'
