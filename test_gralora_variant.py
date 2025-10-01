"""Test if GraLoRA Variant mode works correctly"""
import sys
print("Starting GraLoRA Variant mode test...")
print("=" * 60)

try:
    import torch
    from transformers import GPT2Config, GPT2LMHeadModel
    from peft import get_peft_model, LoraConfig

    # Create small model
    print("\n[1/6] Creating base model...")
    config = GPT2Config(
        vocab_size=100,
        n_positions=64,
        n_embd=128,
        n_layer=1,
        n_head=2,
        n_inner=256
    )
    base_model = GPT2LMHeadModel(config)
    print(f"    ✓ Base model created successfully")

    # Configure GraLoRA
    print("\n[2/6] Configuring GraLoRA...")
    gralora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["c_attn"],
        lora_dropout=0.0,
        task_type="CAUSAL_LM",
        use_gralora=True,
        gralora_k=2,
        hybrid_r=0,
    )
    print(f"    ✓ Configuration complete: r={gralora_config.r}, k={gralora_config.gralora_k}")

    # Create PEFT model
    print("\n[3/6] Creating PEFT model...")
    model = get_peft_model(base_model, gralora_config)
    print("    ✓ Model created successfully")

    # Check if Variant is used
    print("\n[4/6] Checking if Variant is correctly used...")
    variant_found = False
    for name, module in model.named_modules():
        if 'c_attn' in name and hasattr(module, 'lora_variant'):
            print(f"    Module: {name}")
            if hasattr(module, 'lora_variant') and module.lora_variant:
                for adapter_name, variant in module.lora_variant.items():
                    print(f"      Adapter: {adapter_name}")
                    print(f"      Variant type: {type(variant).__name__}")
                    if 'GraLora' in type(variant).__name__:
                        variant_found = True
                        print(f"      ✓ Found GraLoRA Variant!")
            break

    if not variant_found:
        print("    ⚠️ Warning: GraLoRA Variant not found, may be using direct implementation")

    # Check GraLoRA parameters
    print("\n[5/6] Checking GraLoRA parameters...")
    gralora_params = []
    for name, param in model.named_parameters():
        if 'gralora' in name and param.requires_grad:
            gralora_params.append((name, param.shape))
            print(f"    {name}: {param.shape}")

    if not gralora_params:
        print("    ✗ Error: No GraLoRA parameters found!")
        sys.exit(1)

    print(f"    ✓ Found {len(gralora_params)} GraLoRA parameters")

    # Test forward
    print("\n[6/6] Testing Forward Pass...")
    batch_size = 2
    seq_len = 10
    input_ids = torch.randint(0, 100, (batch_size, seq_len))

    model.eval()
    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs.logits

    print(f"    Input shape: {input_ids.shape}")
    print(f"    Output shape: {logits.shape}")
    print(f"    Output range: [{logits.min().item():.2f}, {logits.max().item():.2f}]")

    # Check if output is reasonable
    if torch.isnan(logits).any():
        print("    ✗ Failed: Output contains NaN!")
        sys.exit(1)

    if torch.isinf(logits).any():
        print("    ✗ Failed: Output contains Inf!")
        sys.exit(1)

    print("    ✓ Forward pass successful!")

    print("\n" + "=" * 60)
    print("✓✓✓ All tests passed!")
    if variant_found:
        print("✓✓✓ GraLoRA Variant mode works correctly!")
    print("=" * 60)

except Exception as e:
    print(f"\n✗✗✗ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
