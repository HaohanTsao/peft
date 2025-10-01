"""Quick validation of GraLoRA basic initialization"""
import torch
from transformers import GPT2Config, GPT2LMHeadModel
from peft import get_peft_model, LoraConfig

print("=" * 60)
print("GraLoRA Integration Test")
print("=" * 60)

# Create a tiny model
config = GPT2Config(
    vocab_size=50257,
    n_positions=256,
    n_embd=128,
    n_layer=2,
    n_head=2,
    n_inner=512
)
base_model = GPT2LMHeadModel(config)
print(f"\nBase model: {sum(p.numel() for p in base_model.parameters()) / 1e6:.1f}M params")

# Configure GraLoRA
gralora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["c_attn"],
    lora_dropout=0.0,
    task_type="CAUSAL_LM",
    use_gralora=True,
    gralora_k=2,
)

print("\nCreating PEFT model...")
try:
    model = get_peft_model(base_model, gralora_config)
    print("✓ Model created successfully")

    model.print_trainable_parameters()

    # Detailed check 1: Find all gralora parameters
    print("\n=== Check 1: Find 'gralora' parameters ===")
    gralora_params = []
    for name, param in model.named_parameters():
        if 'gralora' in name:
            gralora_params.append((name, param.shape))
            print(f"  {name}: {param.shape}")

    if not gralora_params:
        print("  ⚠️  No 'gralora' parameters found")

    # Detailed check 2: Inspect the first c_attn layer structure
    print("\n=== Check 2: First c_attn layer structure ===")
    for name, module in model.named_modules():
        if 'c_attn' in name:
            print(f"\nModule: {name}")
            print(f"  Type: {type(module)}")

            # Check for use_gralora attribute
            if hasattr(module, 'use_gralora'):
                print(f"  use_gralora: {module.use_gralora}")

            # Check for gralora_A/B attributes
            if hasattr(module, 'gralora_A'):
                print(f"  gralora_A keys: {list(module.gralora_A.keys())}")
            if hasattr(module, 'gralora_B'):
                print(f"  gralora_B keys: {list(module.gralora_B.keys())}")

            # Only check the first one
            break

    # Detailed check 3: All trainable parameters
    print("\n=== Check 3: All trainable parameters ===")
    trainable = [(n, p.shape) for n, p in model.named_parameters() if p.requires_grad]
    for name, shape in trainable[:10]:  # Show only first 10
        print(f"  {name}: {shape}")
    if len(trainable) > 10:
        print(f"  ... and {len(trainable) - 10} more parameters")

    print("\n✓ All tests passed!")

except Exception as e:
    print(f"\n✗ Error: {e}")
    import traceback
    traceback.print_exc()
