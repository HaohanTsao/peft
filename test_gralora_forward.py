"""Test GraLoRA forward pass"""
import torch
from transformers import GPT2Config, GPT2LMHeadModel, AutoTokenizer
from peft import get_peft_model, LoraConfig

config = GPT2Config(n_embd=128, n_layer=2, n_head=2, n_inner=512)
base_model = GPT2LMHeadModel(config)

gralora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["c_attn"],
    use_gralora=True,
    gralora_k=2,
)

model = get_peft_model(base_model, gralora_config)
tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
tokenizer.pad_token = tokenizer.eos_token

print("Testing Forward Pass...")
try:
    inputs = tokenizer("Hello world", return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    print(f"✓ Forward successful: {outputs.logits.shape}")
except Exception as e:
    print(f"✗ Forward failed: {e}")
    import traceback
    traceback.print_exc()
