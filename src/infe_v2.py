import json
import numpy as np
from PIL import Image
import torch
from transformers import AutoModelForCausalLM, AutoProcessor
from tqdm import tqdm

# Load model and processor
model_id = "/content/Phi3V-Finetuning/Phi-3-vision-128k-instruct"
peft_model_id = "/content/drive/MyDrive/VQA/checkpoint-4866"
img_valid_path = "/content/drive/MyDrive/VQA/valid/"

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="cuda",
    trust_remote_code=True,
    torch_dtype=torch.float16,
    _attn_implementation="flash_attention_2",
)
model.load_adapter(peft_model_id)
processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

# Load the valid.json file
with open("valid_new.json", "r") as f:
    valid_data = json.load(f)

results = []

for item in tqdm(valid_data, desc="Processing images and questions"):
    image_path = img_valid_path + item["image"]
    question = "<|image_1|>\n"+item["question"]
    # print(question)

    # Open the image
    image = Image.open(image_path)

    # Prepare the prompt and inputs
    messages = [{"role": "user", "content": f"{question}"}]
    prompt = processor.tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = processor(prompt, [image], return_tensors="pt").to("cuda")

    # Generate response
    generation_args = {
        "max_new_tokens": 10,
        "top_p": 0.9,
        "temperature": 0.7,
        "do_sample": True,
    }

    generate_ids = model.generate(
        **inputs, eos_token_id=processor.tokenizer.eos_token_id, **generation_args
    )
    # Remove input tokens
    generate_ids = generate_ids[:, inputs["input_ids"].shape[1] :]
    response = processor.batch_decode(
        generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]
    # print(type(response))
    # print(response)
    # Store the response
    results.append(response)

# Convert results to a NumPy array and save to submit.npy
results_array = np.array(results)
np.save("submit_4866_3.npy", results_array)

# print("Processing complete. Results saved to submit.npy.")
