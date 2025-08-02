# Gemma 3.1B FFT Trainer Guide

This guide provides instructions for setting up and running the Gemma 3.1B FFT trainer script.

## 1. Environment Setup

### Kaggle

To use the Gemma model from Kaggle, you need to authenticate. You can do this in two ways:

1.  **Environment Variables**: Set the `KAGGLE_USERNAME` and `KAGGLE_KEY` environment variables with your Kaggle credentials.
2.  **Interactive Login**: If the environment variables are not set, the script will prompt you to log in interactively using `kagglehub.login()`.

### Weights & Biases (W&B)

The script uses Weights & Biases for logging metrics. Go to the W&B website and get your key. Set your ke
```bash
export WANDB_API_KEY=<key>
```

### Hugging Face (HF)

The fine-tuned model is saved in `.safetensors` format, which is compatible with the Hugging Face ecosystem.

## 2. Execution

To run the training script, use the following command, replacing the placeholders with your actual directory paths:

```bash
python gemma3_1b_fft_trainer.py \
    --intermediate_ckpt_dir <path_to_intermediate_checkpoints> \
    --ckpt_dir <path_to_final_checkpoints> \
    --profiling_dir <path_to_profiling_output> \
    --servable_ckpt_dir <path_to_servable_checkpoints> \
    --dataset_name <dataset_name>
```

## 3. Command-Line Arguments

The following command-line arguments are available for the `gemma3_1b_fft_trainer.py` script:

| Argument                  | Purpose                                                 | Required | Default Value                        |
| ------------------------- | ------------------------------------------------------- | -------- | ------------------------------------ |
| `--batch_size`            | Batch size for training.                                | Optional | `8`                                  |
| `--rank`                  | Rank for LoRA.                                          | Optional | `8`                                  |
| `--alpha`                 | Alpha for LoRA.                                         | Optional | `1.0`                                |
| `--max_steps`             | Maximum training steps.                                 | Optional | `200`                                |
| `--eval_every_n_steps`    | Evaluate every N steps.                                 | Optional | `20`                                 |
| `--num_epochs`            | Number of training epochs.                              | Optional | `1`                                  |
| `--dataset_name`          | The name of the dataset to use.                         | Optional | `medalpaca/medical_meadow_medqa`     |
| `--intermediate_ckpt_dir` | Directory to save intermediate checkpoints.             | Required | `./output/intermediate_ckpt/fft/v3/` |
| `--ckpt_dir`              | Directory to save the final model checkpoint.           | Required | `./output/ckpts/medical/fft/v3/`     |
| `--profiling_dir`         | Directory to save profiling output.                     | Required | `./output/profiling/`                |
| `--servable_ckpt_dir`     | Directory to save the servable model checkpoint.        | Required | `./output/servable_ckpt/fft/v1/`     |

## 4. Supported Datasets

The `--dataset_name` argument supports the following datasets:

*   `mtnt/en-fr`
*   `Helsinki-NLP/opus-100`
*   `medalpaca/medical_meadow_medqa`
*   `lavita/ChatDoctor-HealthCareMagic-100k`