# GraLoRA Implementation in PEFT

This directory contains our implementation of GraLoRA (Granular Low-Rank Adaptation) integrated into Hugging Face PEFT.

## Quick Start

### Installation

#### Prerequisites
Make sure you have PyTorch with CUDA support installed (if using GPU):

```bash
# For CUDA 11.8
pip install torch --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1
pip install torch --index-url https://download.pytorch.org/whl/cu121

# For CPU only
pip install torch
```

#### Install PEFT with GraLoRA

```bash
# Clone this fork
git clone https://github.com/HaohanTsao/peft.git
cd peft

# Install in editable mode
pip install -e .

# Install additional dependencies
pip install transformers datasets accelerate
```

### Basic Usage
```python
from transformers import AutoModelForCausalLM
from peft import get_peft_model, LoraConfig

# Load base model
model = AutoModelForCausalLM.from_pretrained("gpt2")

# Configure GraLoRA
config = LoraConfig(
    r=16,                    # Total rank
    lora_alpha=32,
    target_modules=["c_attn", "c_proj"],
    use_gralora=True,        # Enable GraLoRA
    gralora_k=4,             # 4x4 blocks
    hybrid_r=4,              # Hybrid mode: 4 ranks for vanilla LoRA
    task_type="CAUSAL_LM"
)

# Apply PEFT
model = get_peft_model(model, config)
model.print_trainable_parameters()
```

## Test Files

### Core Demonstration
- **`test_gralora_comprehensive.py`**: Complete test suite demonstrating all features
  - Parameter initialization and shapes
  - Block-diagonal structure verification
  - Forward pass with information exchange
  - Merge/unmerge functionality
  - Training compatibility
  - Parameter efficiency comparison

### Fine-tuning Example
- **`example_gralora_finetuning.py`**: Practical fine-tuning example
  - Command-line interface matching original research team's scripts
  - Supports both pure GraLoRA and hybrid mode
  - Compatible with Hugging Face Trainer

## Running Tests

```bash
# Run comprehensive test
python test_gralora_comprehensive.py

# Run fine-tuning example
python example_gralora_finetuning.py --base_model "gpt2" --use_gralora --gralora_k 4
```

## Implementation Details

### Architecture
- **Config**: `src/peft/tuners/lora/config.py` - Added `use_gralora`, `gralora_k`, `hybrid_r`
- **Layer**: `src/peft/tuners/lora/layer.py` - GraLoRA parameter initialization and delta weight computation
- **Variant**: `src/peft/tuners/lora/variants.py` - `GraLoraLinearVariant` class for forward/merge operations
- **Model**: `src/peft/tuners/lora/model.py` - Parameter passing to layer creation

### Key Features
1. ✅ Block-diagonal weight updates (k×k sub-blocks)
2. ✅ Information exchange via tensor permutation
3. ✅ Hybrid mode (GraLoRA + vanilla LoRA)
4. ✅ Merge/unmerge for inference optimization
5. ✅ Full PEFT integration (save/load, quantization, etc.)

### Differences from Original Implementation

| Aspect | Original (`GraLoRA_ref/`) | Our Implementation |
|--------|---------------------------|-------------------|
| Config | `GraloraConfig` | `LoraConfig` with `use_gralora=True` |
| API | Separate PEFT type | Unified with LoRA |
| Features | Basic GraLoRA | + merge, quantization, all PEFT features |
| Pattern | Standalone | LoRA Variant pattern |

## Test Results

All tests pass successfully:

```
✓ Test 1: Parameter creation and shapes
✓ Test 2: Block-diagonal structure
✓ Test 3: Forward pass with information exchange
✓ Test 4: Pure vs Hybrid mode comparison
✓ Test 5: Merge/unmerge functionality
✓ Test 6: Save/load compatibility
✓ Test 7: Training compatibility (gradient computation)
✓ Test 8: Parameter efficiency analysis
```

## Reference

Paper: [GraLoRA: Granular Low-Rank Adaptation for Parameter-Efficient Fine-Tuning](https://arxiv.org/abs/2505.20355)

Original Implementation: See [SqueezeBits/GraLoRA](https://github.com/SqueezeBits/GraLoRA)

## Contact

For questions about this implementation, please open an issue on GitHub.
