import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import argparse

def generate_response(local_model_path: str, prompt: str, max_new_tokens: int = 200) -> str:
    """
    Generates a response using a locally-stored Gemma model.

    Args:
        local_model_path (str): The path to the directory containing the model.
        prompt (str): The user's input prompt.
        max_new_tokens (int): The maximum number of new tokens to generate.

    Returns:
        str: The generated response.
    """
    try:
        # Load the tokenizer from the local directory
        tokenizer = AutoTokenizer.from_pretrained(local_model_path)

        # Load the model from the local directory
        model = AutoModelForCausalLM.from_pretrained(
            local_model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        
        # Format the prompt for the Gemma model
        formatted_prompt = f"<bos><start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n"
        
        # Prepare your prompt and tokenize it
        inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)

        # Run inference
        outputs = model.generate(**inputs, max_new_tokens=max_new_tokens)

        # Decode and return the result
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response

    except Exception as e:
        return f"An error occurred: {e}"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a local Gemma model with a specified prompt.")
    parser.add_argument("--model_path", type=str, help="The local path to the Gemma model directory.")
    parser.add_argument("--prompt", type=str, default="I have a 5 month old baby, what is the common cause of death? ",
                        help="The prompt to use for the model. Defaults to a placeholder if not provided.")
    parser.add_argument("--max_tokens", type=int, default=200,
                        help="The maximum number of new tokens to generate. Defaults to 200.")

    args = parser.parse_args()

    # Call the function with the command-line arguments
    generated_text = generate_response(
        local_model_path=args.model_path,
        prompt=args.prompt,
        max_new_tokens=args.max_tokens
    )
    print(generated_text)
