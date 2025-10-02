"""
Example: Fine-tuning with GraLoRA in PEFT

This example demonstrates how to use GraLoRA for fine-tuning language models,
following the same pattern as the original research team's implementation but
using our PEFT integration.

Usage:
    # Basic usage with default settings
    python example_gralora_finetuning.py --base_model "gpt2" --use_gralora

    # With custom parameters
    python example_gralora_finetuning.py \
        --base_model "meta-llama/Llama-2-7b-hf" \
        --use_gralora \
        --gralora_k 4 \
        --hybrid_r 4 \
        --lora_r 16 \
        --learning_rate 2e-5
"""

import argparse

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
    set_seed,
)

from peft import LoraConfig, get_peft_model


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune with GraLoRA")

    # Model parameters
    parser.add_argument("--base_model", type=str, default="gpt2", help="Base model to fine-tune")
    parser.add_argument("--output_dir", type=str, default="./gralora_output", help="Output directory")

    # Method selection
    parser.add_argument("--use_gralora", action="store_true", help="Use GraLoRA instead of standard LoRA")

    # LoRA/GraLoRA parameters
    parser.add_argument("--lora_r", type=int, default=8, help="LoRA rank (total rank for GraLoRA)")
    parser.add_argument("--lora_alpha", type=int, default=16, help="LoRA alpha (scaling factor)")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout")
    parser.add_argument(
        "--lora_target_modules",
        type=str,
        nargs="+",
        default=["c_attn", "c_proj"],  # For GPT2
        help="Target modules for LoRA",
    )

    # GraLoRA specific parameters
    parser.add_argument("--gralora_k", type=int, default=2, help="Number of blocks (k x k grid)")
    parser.add_argument("--hybrid_r", type=int, default=0, help="Rank for hybrid vanilla LoRA component")

    # Training parameters
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--max_seq_length", type=int, default=256)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=100)
    parser.add_argument("--fp16", action="store_true", help="Use mixed precision")

    return parser.parse_args()


def prepare_dataset(tokenizer, max_length=256):
    """
    Prepare a simple dataset for demonstration.
    In practice, you would use your own dataset.
    """
    # Using a small subset of wikitext for demonstration
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train[:1000]")

    def tokenize_function(examples):
        # Simple causal language modeling: predict next token
        result = tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_length,
            padding="max_length",
            return_tensors=None,
        )
        # For causal LM, labels are the same as input_ids
        result["labels"] = result["input_ids"].copy()
        return result

    tokenized_dataset = dataset.map(
        tokenize_function,
        remove_columns=dataset.column_names,
        batched=True,
    )

    # Filter out empty examples
    tokenized_dataset = tokenized_dataset.filter(lambda x: len(x["input_ids"]) > 0)

    return tokenized_dataset


def main():
    args = parse_args()

    print("=" * 80)
    print("GraLoRA Fine-tuning Example")
    print("=" * 80)
    print()

    # Set seed for reproducibility
    set_seed(args.seed)

    # Load model and tokenizer
    print(f"Loading model: {args.base_model}")
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.float16 if args.fp16 else torch.float32,
        device_map="auto",
    )

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)

    # Set pad token if not present
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Configure PEFT
    if args.use_gralora:
        print("\nConfiguring GraLoRA:")
        print(f"  Total rank (r):           {args.lora_r}")
        print(f"  Blocks (k):               {args.gralora_k}")
        print(f"  Hybrid rank:              {args.hybrid_r}")
        print(f"  GraLoRA rank:             {args.lora_r - args.hybrid_r}")
        print(f"  Rank per block:           {(args.lora_r - args.hybrid_r) // args.gralora_k}")

        peft_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules=args.lora_target_modules,
            lora_dropout=args.lora_dropout,
            task_type="CAUSAL_LM",
            use_gralora=True,
            gralora_k=args.gralora_k,
            hybrid_r=args.hybrid_r,
        )
    else:
        print("\nConfiguring standard LoRA:")
        print(f"  Rank (r):                 {args.lora_r}")

        peft_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules=args.lora_target_modules,
            lora_dropout=args.lora_dropout,
            task_type="CAUSAL_LM",
        )

    # Apply PEFT
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # Prepare dataset
    print("\nPreparing dataset...")
    train_dataset = prepare_dataset(tokenizer, max_length=args.max_seq_length)
    print(f"Training examples: {len(train_dataset)}")

    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        fp16=args.fp16,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=3,
        remove_unused_columns=False,
        seed=args.seed,
    )

    # Data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        padding=True,
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
    )

    # Train
    print("\n" + "=" * 80)
    print("Starting training...")
    print("=" * 80)
    trainer.train()

    # Save
    print("\n" + "=" * 80)
    print("Saving model...")
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Model saved to: {args.output_dir}")

    # Test inference
    print("\n" + "=" * 80)
    print("Testing inference...")
    print("=" * 80)

    model.eval()
    test_prompt = "The quick brown fox"
    inputs = tokenizer(test_prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=20,
            do_sample=True,
            temperature=0.7,
        )

    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Prompt: {test_prompt}")
    print(f"Generated: {generated_text}")

    print("\n" + "=" * 80)
    print("✓ Training complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
