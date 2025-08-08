"""
Trains a Gemma 3.1B model with FFT and LoRA on a specified dataset.

This script handles the entire training pipeline, including data loading,
model configuration with LoRA, training, evaluation, and saving checkpoints.

Usage:
------
To run the training script, use the following command structure.
You must provide the required directory paths for checkpoints and profiling.

Example:
--------
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

Arguments:
----------
    --batch_size (int, optional):
        Batch size for training.
        Default: 8

    --rank (int, optional):
        Rank for LoRA.
        Default: 8

    --alpha (float, optional):
        Alpha for LoRA.
        Default: 1.0

    --max_steps (int, optional):
        Maximum training steps.
        Default: 200

    --eval_every_n_steps (int, optional):
        Evaluate every N steps.
        Default: 20

    --num_epochs (int, optional):
        Number of training epochs.
        Default: 1

    --dataset_name (str, optional):
        The name of the dataset to use from the Hugging Face Hub.
        Default: "medalpaca/medical_meadow_medqa"

    --intermediate_ckpt_dir (str, required):
        Directory to save intermediate checkpoints during training.

    --ckpt_dir (str, required):
        Directory to save the final model checkpoint.

    --profiling_dir (str, required):
        Directory to save profiling output.

    --servable_ckpt_dir (str, required):
        Directory to save the servable model checkpoint.
"""
import os
import shutil
import argparse
from flax import nnx
import jax
import jax.numpy as jnp
import kagglehub
import optax
from orbax import checkpoint as ocp
from qwix import lora
from tunix.generate import sampler as sampler_lib
import data as data_lib
from tunix.models.gemma import gemma as gemma_lib
from tunix.models.gemma3 import model as gemma3_lib
from tunix.models.gemma3 import params as params_lib
from tunix.sft import metrics_logger
from tunix.sft import peft_trainer
import wandb
from safetensors.flax import save_file
from huggingface_hub import snapshot_download

# -- Global Constants --
# Model
MESH = [(1, 4), ("fsdp", "tp")]

def parse_args():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description="Gemma3 1B Medical QA FFT Trainer")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for training.")
    parser.add_argument("--rank", type=int, default=8, help="Rank for LoRA.")
    parser.add_argument("--alpha", type=float, default=1.0, help="Alpha for LoRA.")
    parser.add_argument("--max_steps", type=int, default=200, help="Maximum training steps.")
    parser.add_argument("--eval_every_n_steps", type=int, default=20, help="Evaluate every N steps.")
    parser.add_argument("--num_epochs", type=int, default=1, help="Number of training epochs.")
    parser.add_argument("--dataset_name", type=str, default='medalpaca/medical_meadow_medqa', help="The name of the dataset to use.")
    parser.add_argument("--intermediate_ckpt_dir", type=str, required=True,default="./output/intermediate_ckpt/fft/v3/", help="Intermediate checkpoint directory.")
    parser.add_argument("--ckpt_dir", type=str,required=True, default="./output/ckpts/medical/fft/v3/", help="Checkpoint directory.")
    parser.add_argument("--profiling_dir", type=str, required=True, default="./output/profiling/", help="Profiling directory.")
    parser.add_argument("--servable_ckpt_dir", type=str, required=True, default="./output/servable_ckpt/fft/v1/", help="Servable checkpoint directory.")
    parser.add_argument("--test_prompts_file", type=str, default=None, help="Path to a file containing test prompts, one per line.")
    return parser.parse_args()

def setup_environment(args):
    """Sets up the environment, creating directories and handling Kaggle login."""
    print("Welcome to Corrected Gemma3 1B LoRA Tuning")
    print("Let's have some fun with Gemma3!")
    os.environ["JAX_TRACEBACK_FILTERING"] = "off"

    def chk_mkdir(dir_path):
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

    chk_mkdir(args.intermediate_ckpt_dir)
    chk_mkdir(args.ckpt_dir)
    chk_mkdir(args.profiling_dir)
    chk_mkdir(args.servable_ckpt_dir)

    if "KAGGLE_USERNAME" not in os.environ or "KAGGLE_KEY" not in os.environ:
        kagglehub.login()


def download_model():
    """Downloads the Gemma3 model from Kaggle Hub."""
    print("Downloading Gemma3 1B model...")
    return kagglehub.model_download("google/gemma-3/flax/gemma3-1b-it")


def load_model_and_tokenizer(kaggle_ckpt_path, args):
    """Loads the Gemma3 model and tokenizer."""
    print("Loading Gemma3 model and tokenizer...")
    model_config = gemma3_lib.Gemma3Config.gemma3_1b()
    
    # Load the model with instruction-tuned weights
    it_checkpoint_path = os.path.join(kaggle_ckpt_path, "gemma3-1b-it")
    gemma3 = params_lib.create_model_from_checkpoint(
        checkpoint_path=it_checkpoint_path,
        model_config=model_config,
        mesh=None
    )

    # Save an intermediate checkpoint of the base model
    checkpointer = ocp.StandardCheckpointer()
    _, state = nnx.split(gemma3)
    checkpoint_path = os.path.join(args.intermediate_ckpt_dir, "state")
    if os.path.exists(checkpoint_path):
        shutil.rmtree(checkpoint_path)
    checkpointer.save(os.path.join(checkpoint_path), state)
    checkpointer.wait_until_finished()

    # Load the base model from the intermediate checkpoint
    mesh = jax.make_mesh(*MESH)
    abs_gemma3: nnx.Module = nnx.eval_shape(
        lambda: gemma3_lib.Gemma3(model_config, rngs=nnx.Rngs(params=0))
    )
    abs_state = nnx.state(abs_gemma3)
    abs_state = jax.tree.map(
        lambda a, s: jax.ShapeDtypeStruct(a.shape, jnp.bfloat16, sharding=s),
        abs_state,
        nnx.get_named_sharding(abs_state, mesh),
    )
    restored_params = checkpointer.restore(checkpoint_path, target=abs_state)
    graph_def, _ = nnx.split(abs_gemma3)
    gemma3 = nnx.merge(graph_def, restored_params)

    # Load tokenizer
    tokenizer_path = os.path.join(kaggle_ckpt_path, 'tokenizer.model')
    tokenizer = params_lib.create_tokenizer(tokenizer_path)
    
    return gemma3, tokenizer, mesh, model_config


def run_inference(model, tokenizer, model_config, prompt_text, title="Inference"):
    """Runs inference on the model with the given prompts."""
    print(f"=== {title} ===")
    sampler = sampler_lib.Sampler(
        transformer=model,
        tokenizer=tokenizer,
        cache_config=sampler_lib.CacheConfig(
            cache_size=256,
            num_layers=model_config.num_layers,
            num_kv_heads=model_config.num_kv_heads,
            head_dim=model_config.head_dim,
        ),
    )
    out_data = sampler(
        input_strings=prompt_text,
        total_generation_steps=50,
    )
    for i, (input_string, out_string) in enumerate(zip(prompt_text, out_data.text)):
        print(f"----------------------")
        print(f"Test {i+1}:")
        print(f"Prompt:\n{input_string}")
        print(f"Output:\n{out_string}")


def train_model(model, tokenizer, mesh, args):
    """Trains the model using the specified training configuration."""
    print("Starting model training...")
    train_ds, validation_ds = data_lib.create_datasets(
        dataset_name=args.dataset_name,
        global_batch_size=args.batch_size,
        max_target_length=256,
        num_train_epochs=args.num_epochs,
        tokenizer=tokenizer,
        instruct_tuned=True,  # Use instruction-tuned format for Gemma3-1B-IT
    )

    def gen_model_input_fn(x: peft_trainer.TrainingInput):
        pad_mask = x.input_tokens != tokenizer.pad_id()
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
        eval_every_n_steps=args.eval_every_n_steps,
        max_steps=args.max_steps,
        checkpoint_root_directory=args.ckpt_dir,
        metrics_logging_options=logging_option,
    )
    
    trainer = peft_trainer.PeftTrainer(model, optax.adamw(1e-5), training_config)
    trainer = trainer.with_gen_model_input_fn(gen_model_input_fn)
    
    with mesh:
        trainer.train(train_ds, validation_ds)
    
    return model


def save_safetensors(model, model_config, args):
    """Converts and saves the fine-tuned model to .safetensors format."""
    print("\nConverting and saving the fine-tuned model to .safetensors format...")
    if os.path.exists(args.servable_ckpt_dir):
        shutil.rmtree(args.servable_ckpt_dir)
    os.makedirs(args.servable_ckpt_dir, exist_ok=True)

    snapshot_download(
        repo_id="google/gemma-3-1b-it",
        allow_patterns="*.json",
        local_dir=args.servable_ckpt_dir
    )

    _, state = nnx.split(model)
    
    # The conversion function remains complex and is kept as is.
    def model_state_to_hf_weights(state: nnx.State, model_config, rank: int, alpha: float) -> dict:
        weights_dict = {}
        cpu = jax.devices("cpu")[0]
        scaling_factor = alpha / rank
        vocab_size = 262144  # Assuming vocab size is 262144 for Gemma3 1B
        weights_dict['model.embed_tokens.weight'] = jax.device_put(
            state.embedder.input_embedding.value, cpu
        )[:vocab_size, :]
        weights_dict['model.norm.weight'] = jax.device_put(state.final_norm.scale.value, cpu)
        
        embed_dim = 1152
        hidden_dim = 6 * 1152
        num_heads = 4
        num_kv_heads = 1
        head_dim = 256
        
        for idx, layer in state.layers.items():
            weights_dict[f'model.layers.{idx}.input_layernorm.weight'] = jax.device_put(layer.pre_attention_norm.scale.value, cpu)
            weights_dict[f'model.layers.{idx}.post_attention_layernorm.weight'] = jax.device_put(layer.post_attention_norm.scale.value, cpu)
            weights_dict[f'model.layers.{idx}.self_attn.q_norm.weight'] = jax.device_put(layer.attn._query_norm.scale.value, cpu)
            weights_dict[f'model.layers.{idx}.self_attn.k_norm.weight'] = jax.device_put(layer.attn._key_norm.scale.value, cpu)
            weights_dict[f'model.layers.{idx}.pre_feedforward_layernorm.weight'] =  jax.device_put(layer.pre_ffw_norm.scale.value, cpu)
            weights_dict[f'model.layers.{idx}.post_feedforward_layernorm.weight'] = jax.device_put(layer.post_ffw_norm.scale.value, cpu)
            attn = layer.attn
            base_w_q = attn.q_einsum.w.value.transpose((0, 2, 1)).reshape((num_heads * head_dim, embed_dim))
            weights_dict[f'model.layers.{idx}.self_attn.q_proj.weight'] = jax.device_put(base_w_q, cpu)
            base_w_k = attn.kv_einsum.w.value[0].reshape(embed_dim, -1).T
            weights_dict[f'model.layers.{idx}.self_attn.k_proj.weight'] = jax.device_put(base_w_k, cpu)
            base_w_v = attn.kv_einsum.w.value[1].reshape(embed_dim, -1).T
            weights_dict[f'model.layers.{idx}.self_attn.v_proj.weight'] = jax.device_put(base_w_v, cpu)
            base_w_o = attn.attn_vec_einsum.w.value.reshape(num_heads* head_dim, embed_dim).T
            weights_dict[f'model.layers.{idx}.self_attn.o_proj.weight'] = jax.device_put(base_w_o, cpu)
            mlp = layer.mlp
            base_w_gate = mlp.gate_proj.kernel.value.T
            weights_dict[f'model.layers.{idx}.mlp.gate_proj.weight'] = jax.device_put(base_w_gate, cpu)
            base_w_up = mlp.up_proj.kernel.value.T
            weights_dict[f'model.layers.{idx}.mlp.up_proj.weight'] = jax.device_put(base_w_up, cpu)
            base_w_down = mlp.down_proj.kernel.value.T
            weights_dict[f'model.layers.{idx}.mlp.down_proj.weight'] = jax.device_put(base_w_down, cpu)
        return weights_dict

    hf_weights = model_state_to_hf_weights(state, model_config, args.rank, args.alpha)
    
    # Safetensors saving logic would follow here.
    # For now, we just print a success message.
    print("Model successfully converted to Hugging Face format.")
    # In a real scenario, you would use a library like `safetensors.torch.save_file`
    # to save the `hf_weights` dictionary.
    # from safetensors.torch import save_file
    save_file(hf_weights, os.path.join(args.servable_ckpt_dir, "model.safetensors"))
    print(f"Safetensors file would be saved to: {os.path.join(args.servable_ckpt_dir, 'model.safetensors')}")


def format_instruction_prompt(prompt):
    """Format prompt using the official Gemma3-1B-IT chat template."""
    return f"<start_of_turn>user\n{prompt}\n<end_of_turn>\n<start_of_turn>model\n"


def main(args):
    """Orchestrates the entire model training and evaluation workflow."""
    setup_environment(args)
    kaggle_path = download_model()
    model, tokenizer, mesh, model_config = load_model_and_tokenizer(kaggle_path, args)

    # Initial inference before training
    if args.test_prompts_file:
        with open(args.test_prompts_file, 'r') as f:
            test_prompts = [format_instruction_prompt(line.strip()) for line in f]
    else:
        test_prompts = [
            format_instruction_prompt("What are the symptoms of pneumonia?"),
            format_instruction_prompt("How is hypertension diagnosed?"),
            format_instruction_prompt("What is the procedure for a colonoscopy?"),
            format_instruction_prompt("What are the side effects of aspirin?"),
            format_instruction_prompt("What is the difference between systolic and diastolic blood pressure?"),
        ]
    run_inference(model, tokenizer, model_config, test_prompts, "Base Model Performance")

    # Train the model
    # Initialize wandb
    
    trained_model = train_model(model, tokenizer, mesh, args)
    wandb.init()
    # Inference after training
    run_inference(trained_model, tokenizer, model_config, test_prompts, "Fine-Tuned Model Performance")
    
  

    # Save the final model
    save_safetensors(trained_model, model_config, args)

    print("Script execution complete.")


if __name__ == "__main__":
    args = parse_args()
    main(args)
