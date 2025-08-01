# -*- coding: utf-8 -*-
"""
Gemma3 1B Model Fine-Tuning Script.

This script provides a comprehensive workflow for fine-tuning the Gemma3 1B model.
It includes functionalities for:
- Loading the base model from KaggleHub.
- Setting up training and evaluation datasets.
- Performing full-weight fine-tuning.
- Evaluating the model before and after training.
- Saving the final fine-tuned model in SafeTensors format for serving.
- Command-line interface for easy configuration of hyperparameters and paths.

Example Usage:
python gemma3_finetune.py \
    --batch_size 8 \
    --max_steps 800 \
    --eval_steps 20 \
    --dataset_name "allenai/open_math_2_50k_r1-original" \
    --intermediate_ckpt_dir "/mnt/disks/workdir/gemma3/1b/math/intermediate_ckpt/fft/v3/" \
    --ckpt_dir "/mnt/disks/workdir/gemma3/1b/ckpts/allenai_Math/fft/v3/" \
    --servable_ckpt_dir "/mnt/disks/workdir/final_models/gemma3_1b_allenai_math/fft/v1/"
"""

import argparse
import gc
import json
import os
import shutil
import time

import jax
import jax.numpy as jnp
import kagglehub
import optax
# import wandb
from flax import nnx
from huggingface_hub import snapshot_download
from orbax import checkpoint as ocp
from safetensors.flax import save_file

# Local/project-specific imports
import data as data_lib
from qwix import lora
from tunix.generate import sampler as sampler_lib
from tunix.models.gemma import gemma as gemma_lib
from tunix.models.gemma3 import model as gemma3_lib
from tunix.models.gemma3 import params as params_lib
from tunix.sft import metrics_logger, peft_trainer

# Suppress excessive JAX traceback filtering
os.environ["JAX_TRACEBACK_FILTERING"] = "off"


# --- Utility Functions ---

def chk_mkdir(dir_path: str):
    """Create a directory if it doesn't exist."""
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def cleanup_checkpoints(ckpt_dir: str):
    """Clean up existing checkpoints to avoid parameter conflicts."""
    if os.path.exists(ckpt_dir):
        print(f"Removing existing checkpoint directory: {ckpt_dir}")
        shutil.rmtree(ckpt_dir)
    os.makedirs(ckpt_dir, exist_ok=True)


# --- Model Loading and Preparation ---

def download_and_load_base_model(model_handle: str, model_config: gemma3_lib.Gemma3Config, intermediate_ckpt_dir: str):
    """Download model from KaggleHub, load weights, and save an intermediate checkpoint."""
    print(f"Downloading base model from KaggleHub: {model_handle}")
    if "KAGGLE_USERNAME" not in os.environ or "KAGGLE_KEY" not in os.environ:
        kagglehub.login()
    kaggle_ckpt_path = kagglehub.model_download(model_handle)

    print(f"Loading instruction-tuned weights from: {kaggle_ckpt_path}")
    gemma3 = params_lib.create_model_from_checkpoint(
        checkpoint_path=os.path.join(kaggle_ckpt_path, "gemma3-1b"),
        model_config=model_config,
        mesh=None  # We will shard it later
    )

    # Save an intermediate checkpoint to be loaded in the sharded context
    checkpointer = ocp.StandardCheckpointer()
    _, state = nnx.split(gemma3)
    checkpoint_path = os.path.join(intermediate_ckpt_dir, "state")
    
    if os.path.exists(checkpoint_path):
        shutil.rmtree(checkpoint_path)
    chk_mkdir(checkpoint_path)

    print(f"Saving intermediate base model checkpoint to: {checkpoint_path}")
    checkpointer.save(checkpoint_path, state)
    checkpointer.wait_until_finished()
    
    # Return path to the tokenizer for later use
    tokenizer_path = os.path.join(kaggle_ckpt_path, 'tokenizer.model')
    return tokenizer_path


def get_sharded_model_from_ckpt(ckpt_path: str, model_config: gemma3_lib.Gemma3Config, mesh: jax.sharding.Mesh):
    """Load the base Gemma3 model from a checkpoint and shard it across the mesh."""
    print("Loading and sharding base model from intermediate checkpoint...")
    abs_gemma3: nnx.Module = nnx.eval_shape(
        lambda: gemma3_lib.Gemma3(model_config, rngs=nnx.Rngs(params=0))
    )
    abs_state = nnx.state(abs_gemma3)
    abs_state = jax.tree.map(
        lambda a, s: jax.ShapeDtypeStruct(a.shape, jnp.bfloat16, sharding=s),
        abs_state,
        nnx.get_named_sharding(abs_state, mesh),
    )
    
    checkpointer = ocp.StandardCheckpointer()
    restored_params = checkpointer.restore(ckpt_path, target=abs_state)

    graph_def, _ = nnx.split(abs_gemma3)
    gemma3 = nnx.merge(graph_def, restored_params)
    print("Model successfully loaded and sharded.")
    return gemma3


def evaluate_model(sampler, title: str):
    """Run inference on a batch of test prompts and print the output."""
    input_batch = [
        "Lennon is a sales rep and is paid $0.36 in mileage reimbursement when he travels to meet with clients. On Monday he drove 18 miles. Tuesday he drove 26 miles. Wednesday and Thursday he drove 20 miles each day and on Friday he drove 16 miles. How much money will he be reimbursed?",
        "What is 52 multiplied by 29",
        "Jerry has three times as many stickers as George. George has 6 fewer stickers than his brother Fred. If Fred has 18 stickers, how many stickers does Jerry have?",
        "Josh went to the shopping center. He bought 9 films and 4 books. He also bought 6 CDs. Each film cost $5, each book cost $4 and each CD cost $3. How much did Josh spend in all?",
    ]
    
    print(f"\n=== {title} ===")
    out_data = sampler(
        input_strings=input_batch,
        total_generation_steps=50,
    )

    for i, (input_string, out_string) in enumerate(zip(input_batch, out_data.text)):
        print("----------------------")
        print(f"Test {i+1}:")
        print(f"Prompt:\n{input_string}")
        print(f"Output:\n{out_string}")
    print("=========================\n")


# --- Model Conversion and Saving ---

def model_state_to_hf_weights(state: nnx.State) -> dict:
    """
    Converts a JAX/NNX Gemma state to a Hugging Face compatible weight dictionary.
    This version is for a fully fine-tuned model (no LoRA merging).
    """
    print("Converting fine-tuned JAX state to Hugging Face format...")
    weights_dict = {}
    cpu = jax.devices("cpu")[0]

    # Handle the embedding and final normalization weights
    vocab_size = 262144  # Standard Gemma vocab size
    weights_dict['model.embed_tokens.weight'] = jax.device_put(
        state.embedder.input_embedding.value, cpu
    )[:vocab_size, :]
    weights_dict['model.norm.weight'] = jax.device_put(state.final_norm.scale.value, cpu)

    # Gemma3 1B model dimensions
    embed_dim = 1152
    num_heads = 4
    head_dim = 256
    
    # Dump state structure to a file for debugging if needed
    with open("state_dump.txt", "w", encoding="utf-8") as file:
        file.write(str(state))

    # Iterate through each layer to extract weights
    for idx, layer in state.layers.items():
        weights_dict[f'model.layers.{idx}.input_layernorm.weight'] = jax.device_put(layer.pre_attention_norm.scale.value, cpu)
        weights_dict[f'model.layers.{idx}.post_attention_layernorm.weight'] = jax.device_put(layer.post_attention_norm.scale.value, cpu)
        weights_dict[f'model.layers.{idx}.self_attn.q_norm.weight'] = jax.device_put(layer.attn._query_norm.scale.value, cpu)
        weights_dict[f'model.layers.{idx}.self_attn.k_norm.weight'] = jax.device_put(layer.attn._key_norm.scale.value, cpu)
        weights_dict[f'model.layers.{idx}.pre_feedforward_layernorm.weight'] =  jax.device_put(layer.pre_ffw_norm.scale.value, cpu)
        weights_dict[f'model.layers.{idx}.post_feedforward_layernorm.weight'] = jax.device_put(layer.post_ffw_norm.scale.value, cpu)

        # Attention Block
        attn = layer.attn
        weights_dict[f'model.layers.{idx}.self_attn.q_proj.weight'] = jax.device_put(attn.q_einsum.w.value.transpose((0, 2, 1)).reshape((num_heads * head_dim, embed_dim)), cpu)
        weights_dict[f'model.layers.{idx}.self_attn.k_proj.weight'] = jax.device_put(attn.kv_einsum.w.value[0].reshape(embed_dim, -1).T, cpu)
        weights_dict[f'model.layers.{idx}.self_attn.v_proj.weight'] = jax.device_put(attn.kv_einsum.w.value[1].reshape(embed_dim, -1).T, cpu)
        weights_dict[f'model.layers.{idx}.self_attn.o_proj.weight'] = jax.device_put(attn.attn_vec_einsum.w.value.reshape(num_heads * head_dim, embed_dim).T, cpu)

        # MLP Block
        mlp = layer.mlp
        weights_dict[f'model.layers.{idx}.mlp.gate_proj.weight'] = jax.device_put(mlp.gate_proj.kernel.value.T, cpu)
        weights_dict[f'model.layers.{idx}.mlp.up_proj.weight'] = jax.device_put(mlp.up_proj.kernel.value.T, cpu)
        weights_dict[f'model.layers.{idx}.mlp.down_proj.weight'] = jax.device_put(mlp.down_proj.kernel.value.T, cpu)

    return weights_dict


def flatten_weight_dict(torch_params, prefix=""):
    """Flattens a nested weight dictionary for SafeTensors compatibility."""
    flat_params = {}
    for key, value in torch_params.items():
        new_key = f"{prefix}{key}" if prefix else key
        if isinstance(value, dict):
            flat_params.update(flatten_weight_dict(value, new_key + "."))
        else:
            flat_params[new_key] = value
    return flat_params

# --- Main Execution ---

def main(args):
    """Main function to run the fine-tuning pipeline."""
    print("Welcome to the Gemma3 1B Fine-Tuning Script")

    # --- Setup ---
    chk_mkdir(args.intermediate_ckpt_dir)
    chk_mkdir(args.ckpt_dir)
    chk_mkdir(args.profiling_dir)

    # Define model and mesh configurations
    model_config = gemma3_lib.Gemma3Config.gemma3_1b()
    mesh_shape = (2, 4)
    mesh_axes = ("fsdp", "tp")
    mesh = jax.make_mesh(*mesh_shape, axis_names=mesh_axes)
    
    print(f"Using Device Mesh: {mesh_shape} with axes {mesh_axes}")
    print(f"Training Config: Batch={args.batch_size}, Steps={args.max_steps}, Epochs={args.num_epochs}, Eval Steps={args.eval_steps}")
    if args.use_lora:
        print(f"LoRA Config: Rank={args.lora_rank}, Alpha={args.lora_alpha}")

    # --- Load Base Model ---
    tokenizer_path = download_and_load_base_model(
        args.model_handle, model_config, args.intermediate_ckpt_dir
    )
    
    gemma3, mesh, model_config = get_sharded_model_from_ckpt(
        ckpt_path=os.path.join(args.intermediate_ckpt_dir, "state"),
        model_config=model_config,
        mesh=mesh
    )
    
    gemma3_tokenizer = params_lib.create_tokenizer(tokenizer_path)

    # --- Evaluate Base Model ---
    sampler = sampler_lib.Sampler(
        transformer=gemma3,
        tokenizer=gemma3_tokenizer,
        cache_config=sampler_lib.CacheConfig(
            cache_size=256,
            num_layers=model_config.num_layers,
            num_kv_heads=model_config.num_kv_heads,
            head_dim=model_config.head_dim,
        ),
    )
    evaluate_model(sampler, "Base Model Performance")
    
    # Clean up to free memory before training
    del sampler
    gc.collect()

    # --- Prepare for Training ---
    train_ds, validation_ds = data_lib.create_datasets(
        dataset_name=args.dataset_name,
        global_batch_size=args.batch_size,
        max_target_length=256,
        num_train_epochs=args.num_epochs,
        tokenizer=gemma3_tokenizer,
    )
    
    def gen_model_input_fn(x: peft_trainer.TrainingInput):
        """Generate model inputs from training data."""
        pad_mask = x.input_tokens != gemma3_tokenizer.pad_id()
        positions = gemma_lib.build_positions_from_mask(pad_mask)
        attention_mask = gemma_lib.make_causal_attn_mask(pad_mask)
        return {
            'input_tokens': x.input_tokens,
            'input_mask': x.input_mask,
            'positions': positions,
            'attention_mask': attention_mask,
        }

    logging_option = metrics_logger.MetricsLoggerOptions(
        log_dir=args.profiling_dir, flush_every_n_steps=20
    )

    training_config = peft_trainer.TrainingConfig(
        eval_every_n_steps=args.eval_steps,
        max_steps=args.max_steps,
        checkpoint_root_directory=args.ckpt_dir,
        metrics_logging_options=logging_option,
    )
    
    # --- Training ---
    # NOTE: The original script contained commented-out LoRA code.
    # The `use_lora` flag can be used to switch between full tuning and LoRA.
    if args.use_lora:
        # TODO: Implement the LoRA application logic as in the original script if needed.
        # This part is left as a placeholder based on the original commented code.
        print("LoRA training is selected but not fully implemented in this refactor.")
        # lora_gemma3 = get_lora_model(gemma3, mesh, args.lora_rank, args.lora_alpha)
        # trainer = peft_trainer.PeftTrainer(
        #     lora_gemma3, optax.adamw(1e-3), training_config
        # ).with_gen_model_input_fn(gen_model_input_fn)
    else:
        print("Starting full-weight fine-tuning...")
        trainer = peft_trainer.PeftTrainer(gemma3, optax.adamw(1e-5), training_config)
        trainer = trainer.with_gen_model_input_fn(gen_model_input_fn)

    # Uncomment to use wandb
    # wandb.init(project="gemma3-finetuning", config=vars(args))

    with mesh:
        trainer.train(train_ds, validation_ds)

    print("Training complete.")

    # --- Evaluate Fine-Tuned Model ---
    sampler = sampler_lib.Sampler(
        transformer=gemma3,  # gemma3 is updated in-place by the trainer
        tokenizer=gemma3_tokenizer,
        cache_config=sampler_lib.CacheConfig(
            cache_size=256,
            num_layers=model_config.num_layers,
            num_kv_heads=model_config.num_kv_heads,
            head_dim=model_config.head_dim,
        ),
    )
    evaluate_model(sampler, "Fine-Tuned Model Performance")

    # --- Save Servable Model ---
    print("\nConverting and saving the fine-tuned model to .safetensors format...")
    if os.path.exists(args.servable_ckpt_dir):
        shutil.rmtree(args.servable_ckpt_dir)
    os.makedirs(args.servable_ckpt_dir, exist_ok=True)

    _, state = nnx.split(gemma3)

    with jax.default_device(jax.devices("cpu")[0]):
        hf_weights = model_state_to_hf_weights(state)
    
    servable_weights = flatten_weight_dict(hf_weights)
    save_file(servable_weights, os.path.join(args.servable_ckpt_dir, 'model.safetensors'))
    print(f"Model successfully saved to {os.path.join(args.servable_ckpt_dir, 'model.safetensors')}")

    # Download supplementary files like config.json
    snapshot_download(repo_id="google/gemma-3-1b-it", allow_patterns="*.json", local_dir=args.servable_ckpt_dir)
    print("Downloaded supplementary model config files.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gemma3 1B Fine-Tuning Script")

    # Data and Model
    parser.add_argument("--dataset_name", type=str, default="allenai/open_math_2_50k_r1-original", help="Name of the dataset from Hugging Face Hub.")
    parser.add_argument("--model_handle", type=str, default="google/gemma-3/flax/gemma3-1b/1", help="KaggleHub model handle.")

    # Training Hyperparameters
    parser.add_argument("--batch_size", type=int, default=8, help="Global batch size for training.")
    parser.add_argument("--max_steps", type=int, default=800, help="Maximum number of training steps.")
    parser.add_argument("--eval_steps", type=int, default=20, help="Evaluate every N steps.")
    parser.add_argument("--num_epochs", type=int, default=1, help="Number of training epochs.")

    # LoRA Hyperparameters
    parser.add_argument("--use_lora", action="store_true", help="Enable LoRA for fine-tuning.")
    parser.add_argument("--lora_rank", type=int, default=8, help="Rank for LoRA.")
    parser.add_argument("--lora_alpha", type=float, default=1.0, help="Alpha for LoRA.")

    # Directory Paths
    parser.add_argument("--intermediate_ckpt_dir", type=str, required=True, help="Directory for intermediate model checkpoints.")
    parser.add_argument("--ckpt_dir", type=str, required=True, help="Directory for saving training checkpoints.")
    parser.add_argument("--profiling_dir", type=str, default="/mnt/disks/workdir/gemma3/1b/profiling/", help="Directory for profiling logs.")
    parser.add_argument("--servable_ckpt_dir", type=str, required=True, help="Directory to save the final servable model in SafeTensors format.")
    
    cli_args = parser.parse_args()
    main(cli_args)

