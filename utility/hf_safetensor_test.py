# 1. Make sure you have the necessary libraries installed
# !pip install transformers torch accelerate

from transformers import pipeline
import torch


from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# 2. Define the path to your local model directory
local_model_path = "/home/jupyter/gemma_safe_tensor_ckpts"

# 3. Load the tokenizer from the local directory
tokenizer = AutoTokenizer.from_pretrained(local_model_path)

# 4. Load the model from the local directory
#    This will automatically find and use the model.safetensors file.
model = AutoModelForCausalLM.from_pretrained(
    local_model_path,
    torch_dtype=torch.bfloat16, # Adjust dtype as needed
    device_map="auto"           # Automatically use GPU if available
)

# 5. Prepare your prompt and tokenize it
#prompt = "I have a 5 month old baby, what is the common cause of death? "
prompt = "<bos><start_of_turn>user\nI have a 5 month old baby, what is the common cause of death?<end_of_turn>\n<start_of_turn>model\n"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

# 6. Run inference
outputs = model.generate(**inputs, max_new_tokens=200)

# 7. Decode and print the result
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
