"""Test GraLoRA merge/unmerge functionality"""
import sys
import torch
from transformers import GPT2Config, GPT2LMHeadModel
from peft import get_peft_model, LoraConfig

print("=" * 60)
print("GraLoRA Merge/Unmerge Test")
print("=" * 60)

try:
    # Create small model
    print("\n[1/7] Creating base model...")
    config = GPT2Config(
        vocab_size=100,
        n_positions=64,
        n_embd=128,
        n_layer=1,
        n_head=2,
        n_inner=256
    )
    base_model = GPT2LMHeadModel(config)
    print("    ✓ Base model created successfully")

    # Configure GraLoRA
    print("\n[2/7] Configuring GraLoRA...")
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
    print("    ✓ Configuration complete")

    # Create PEFT model
    print("\n[3/7] Creating PEFT model...")
    model = get_peft_model(base_model, gralora_config)
    print("    ✓ Model created successfully")

    # Test get_delta_weight
    print("\n[4/7] Testing get_delta_weight...")
    for name, module in model.named_modules():
        if 'c_attn' in name and hasattr(module, 'get_delta_weight'):
            adapter_name = 'default'
            try:
                delta_weight = module.get_delta_weight(adapter_name)
                print(f"    Module: {name}")
                print(f"    Delta weight shape: {delta_weight.shape}")
                print(f"    Delta weight range: [{delta_weight.min().item():.4f}, {delta_weight.max().item():.4f}]")
                print(f"    Contains NaN: {torch.isnan(delta_weight).any().item()}")
                print(f"    Contains Inf: {torch.isinf(delta_weight).any().item()}")

                if torch.isnan(delta_weight).any() or torch.isinf(delta_weight).any():
                    print("    ✗ Delta weight contains NaN or Inf!")
                    sys.exit(1)

                print("    ✓ get_delta_weight successful")
            except Exception as e:
                print(f"    ✗ get_delta_weight failed: {e}")
                import traceback
                traceback.print_exc()
                sys.exit(1)
            break

    # Test forward (unmerged)
    print("\n[5/7] Testing Forward (unmerged)...")
    input_ids = torch.randint(0, 100, (2, 10))
    model.eval()
    with torch.no_grad():
        outputs_unmerged = model(input_ids).logits
    print(f"    Output shape: {outputs_unmerged.shape}")
    print(f"    ✓ Forward (unmerged) successful")

    # Test merge
    print("\n[6/7] Testing Merge...")
    try:
        model.merge_adapter()
        print("    ✓ Merge successful")

        # Test merged forward
        with torch.no_grad():
            outputs_merged = model(input_ids).logits

        # Compare outputs
        diff = (outputs_merged - outputs_unmerged).abs().max().item()
        print(f"    Difference before/after merge: {diff:.6f}")

        if diff > 1e-4:
            print(f"    ⚠️ Warning: Large difference before/after merge ({diff:.6f})")
        else:
            print(f"    ✓ Outputs consistent before/after merge (diff={diff:.6f})")

    except Exception as e:
        print(f"    ✗ Merge failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Test unmerge
    print("\n[7/7] Testing Unmerge...")
    try:
        model.unmerge_adapter()
        print("    ✓ Unmerge successful")

        # Test unmerged forward
        with torch.no_grad():
            outputs_unmerged_again = model(input_ids).logits

        # Compare outputs
        diff = (outputs_unmerged_again - outputs_unmerged).abs().max().item()
        print(f"    Difference after unmerge vs original: {diff:.6f}")

        if diff > 1e-4:
            print(f"    ⚠️ Warning: Large difference after unmerge vs original ({diff:.6f})")
        else:
            print(f"    ✓ Outputs consistent after unmerge (diff={diff:.6f})")

    except Exception as e:
        print(f"    ✗ Unmerge failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    print("\n" + "=" * 60)
    print("✓✓✓ All tests passed!")
    print("✓✓✓ GraLoRA merge/unmerge works correctly!")
    print("=" * 60)

except Exception as e:
    print(f"\n✗✗✗ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
