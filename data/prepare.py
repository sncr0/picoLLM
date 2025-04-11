import tiktoken
from .dataset import MixedSequenceDataset
from datasets import load_dataset

def load_and_prepare_data(args, block_size, train_subset_size):
    """
    Load and prepare data from TinyStories and custom input files
    
    Args:
        args: Parsed command line arguments
        block_size: Maximum sequence length
        train_subset_size: Number of samples to use from TinyStories
        
    Returns:
        combined_dataset: The complete dataset
        enc: The tokenizer
        vocab_size: Size of the vocabulary
    """
    # Initialize tokenizer
    enc = tiktoken.get_encoding("gpt2")
    vocab_size = enc.n_vocab
    print(f"Vocab size: {vocab_size}")
    
    # Load TinyStories if weight > 0
    tinystories_seqs = []
    if args.tinystories_weight > 0.0:
        print(f"Loading TinyStories from huggingface with weight={args.tinystories_weight}...")
        dataset = load_dataset("roneneldan/TinyStories", split="train")
        dataset = dataset.select(range(train_subset_size))
        
        for sample in dataset:
            text = sample['text']
            tokens = enc.encode(text)
            tokens = tokens[:block_size]
            if len(tokens) > 0:
                tinystories_seqs.append(tokens)
        print(f"TinyStories sequences: {len(tinystories_seqs)}")
    else:
        print("TinyStories weight=0 => skipping TinyStories.")
    
    # Load custom input files
    other_seqs = []
    if args.input_files:
        for filepath in args.input_files:
            print(f"Reading custom text file: {filepath}")
            with open(filepath, "r", encoding="utf-8") as f:
                lines = f.readlines()
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                tokens = enc.encode(line)
                tokens = tokens[:block_size]
                if len(tokens) > 0:
                    other_seqs.append(tokens)
        print(f"Custom input files: {len(other_seqs)} sequences loaded.")
    else:
        print("No custom input files provided.")
    
    # Check data validity
    if len(tinystories_seqs) == 0 and args.tinystories_weight > 0:
        print("Warning: TinyStories is empty but tinystories_weight>0. That's okay, no data from it.")
    
    # Create combined dataset
    combined_dataset = MixedSequenceDataset(
        tinystories_seqs=tinystories_seqs,
        other_seqs=other_seqs,
        p_tiny=args.tinystories_weight
    )
    
    return combined_dataset, enc, vocab_size
