# -*- coding: utf-8 -*-
"""
Zero-shot text reconstruction using Mistral
Analyzes JSONL input texts and reconstructs implicit parts
"""

import os
import re
import json
import torch
import argparse
import random
import numpy as np
from tqdm import tqdm
from huggingface_hub import login
from transformers import (
    Mistral3ForConditionalGeneration,
    BitsAndBytesConfig
)
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from mistral_common.protocol.instruct.request import ChatCompletionRequest

import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    filename="mistral_zero_generation.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def ensure_huggingface_token():
    token = os.getenv("HUGGINGFACE_HUB_TOKEN")
    if not token:
        raise ValueError("Hugging Face token not found. Please ensure it is set in the environment.")
    else:
        logging.info("Hugging Face token found. Logging in...")
        login(token=token)

ensure_huggingface_token()

def set_seed(seed):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def load_jsonl_dataset(path):
    """Load JSONL dataset with input texts"""
    texts = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            ex = json.loads(line)
            # Only load input field for reconstruction
            input_text = ex.get("input", "").strip()
            if input_text:
                texts.append({
                    "input": input_text,
                    "original_output": ex.get("output", "")  # Keep for optional comparison
                })
    return texts

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Text reconstruction using Mistral")
    parser.add_argument(
        "--input_file",
        type=str,
        required=True,
        help="Path to input JSONL file"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./reconstructed_outputs",
        help="Directory to save reconstructed outputs"
    )
    parser.add_argument(
        "--model_id",
        type=str,
        default="mistralai/Mistral-Small-3.2-24B-Instruct-2506",
        help="Hugging Face model ID"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of examples to process (for testing)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=2048,
        help="Maximum number of tokens to generate"
    )
    return parser.parse_args()

def setup_model(model_id):
    """Setup Mistral model with 4-bit quantization"""
    logging.info(f"Loading model: {model_id}")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )
    tokenizer = MistralTokenizer.from_hf_hub(model_id)
    model = Mistral3ForConditionalGeneration.from_pretrained(
        model_id, device_map="auto", quantization_config=bnb_config
    )
    logging.info("Model loaded successfully")
    return model, tokenizer

def build_prompt(text):
    """Build prompt for text reconstruction"""
    prompt = f"""Your task is to analyze the given text and reconstruct implicit parts of the text. The text is argumentative, so implicit parts can be premises or conclusions that are not explicitly stated but are necessary for the argument to hold.

As an output, provide a complete list of all sentences in the text, including both explicit and reconstructed implicit sentences. Mark each sentence with either <Explicit> or <Implicit> tags.

Text:
{text}

Instructions:
- Identify all explicit sentences that are already present in the text
- Identify and reconstruct any implicit premises or conclusions
- Mark each sentence with <Explicit> or <Implicit> tags
- Maintain the logical flow of the argument

Output:
"""
    return prompt

def save_predictions(predictions, output_dir):
    """Save reconstructed texts to JSONL file"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(output_dir, f"reconstructed_texts_{timestamp}.jsonl")
    
    with open(output_file, "w", encoding="utf-8") as f:
        for pred in predictions:
            f.write(json.dumps(pred, ensure_ascii=False) + "\n")
    
    logging.info(f"Reconstructed texts saved to {output_file}")
    return output_file

def generate_reconstruction(model, tokenizer, text, max_new_tokens=2048):
    """Generate reconstruction for a single text"""
    prompt = build_prompt(text)
    
    messages = [{"role": "user", "content": prompt}]
    chat_request = ChatCompletionRequest(messages=messages)
    tokenized = tokenizer.encode_chat_completion(chat_request)
    input_ids = torch.tensor([tokenized.tokens], device=model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False
        )
    
    generated_tokens = outputs[0][len(tokenized.tokens):]
    reconstruction = tokenizer.decode(generated_tokens).strip()
    
    return reconstruction

def main():
    """Main function to run text reconstruction"""
    # Parse arguments
    args = parse_args()
    
    # Setup logging
    os.makedirs(args.output_dir, exist_ok=True)
    log_file = os.path.join(args.output_dir, f"reconstruction_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
    
    logging.info("="*50)
    logging.info("Starting text reconstruction process")
    logging.info(f"Input file: {args.input_file}")
    logging.info(f"Output directory: {args.output_dir}")
    logging.info(f"Model: {args.model_id}")
    logging.info(f"Seed: {args.seed}")
    logging.info("="*50)
    
    # Set seed for reproducibility
    set_seed(args.seed)
    
    # Load data
    logging.info("Loading dataset...")
    data = load_jsonl_dataset(args.input_file)
    logging.info(f"Loaded {len(data)} examples")
    
    # Limit if specified
    if args.limit:
        data = data[:args.limit]
        logging.info(f"Limited to first {args.limit} examples")
    
    # Load model and tokenizer
    logging.info("Loading model...")
    model, tokenizer = setup_model(args.model_id)
    model.eval()
    
    # Process each text
    logging.info(f"Processing {len(data)} texts...")
    predictions = []
    for idx, item in enumerate(tqdm(data, desc="Reconstructing texts")):
        try:
            reconstruction = generate_reconstruction(
                model, 
                tokenizer, 
                item["input"],
                max_new_tokens=args.max_new_tokens
            )
            
            predictions.append({
                "index": idx,
                "original_text": item["input"],
                "reconstructed_text": reconstruction,
                "original_output": item.get("original_output", "")
            })
            
            # Log progress every 10 examples
            if (idx + 1) % 10 == 0:
                logging.info(f"Processed {idx + 1}/{len(data)} examples")
                
        except Exception as e:
            logging.error(f"Error processing example {idx}: {str(e)}")
            predictions.append({
                "index": idx,
                "original_text": item["input"],
                "reconstructed_text": f"ERROR: {str(e)}",
                "original_output": item.get("original_output", "")
            })
    
    # Save results
    output_file = save_predictions(predictions, args.output_dir)
    
    logging.info("="*50)
    logging.info(f"Reconstruction complete! Processed {len(predictions)} examples")
    logging.info(f"Results saved to: {output_file}")
    logging.info(f"Log file: {log_file}")
    logging.info("="*50)

if __name__ == "__main__":
    main()