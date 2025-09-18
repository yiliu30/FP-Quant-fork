from transformers import AutoModelForCausalLM, AutoTokenizer, FPQuantConfig
import torch

torch.set_default_device("cuda")
model_id = "/models/Qwen3-0.6B"
model_id = "/data5/yliu7/HF_HOME/ISTA-DASLab/Llama-3.1-8B-Instruct-FPQuant-GPTQ-MXFP4/"
model = AutoModelForCausalLM.from_pretrained(
    # "qwen/Qwen3-8B",
    model_id,
    # quantization_config=FPQuantConfig(pseudoquantization=True),
    # device_map="auto",
    dtype=torch.bfloat16,
)
tokenizer = AutoTokenizer.from_pretrained(
    # "qwen/Qwen3-8B",
    model_id,
)
breakpoint()

@torch.no_grad()
def batch_gen_text(model, tokenizer, msg="", prompt=["What's AI?"], max_tokens = 50, device="cuda"):
    model = model.to(device)
    inputs = tokenizer.batch_encode_plus(prompt, return_tensors="pt", truncation=True)
    new_tokens = model.generate(**inputs.to(device), max_length=max_tokens)
    text = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
    for i, t in enumerate(text):
        print(f"Generated text ({msg}): {t}")

batch_gen_text(model, tokenizer)