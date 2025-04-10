import argparse
import random
import torch
import matplotlib.pyplot as plt

from data import load_and_prepare_data, seq_collate_fn
from training import train_one_model, generate_text
from models import KGramMLPSeqModel, LSTMSeqModel, TransformerModel, KGramEmbeddingSeqModel
from analysis import monosemantic_analysis_for_token
from config import parse_args

################################################################################
# 9. Main
################################################################################

def main():
    args = parse_args()

    # Additional local variables from arguments
    k = args.kgram_k
    chunk_size = args.kgram_chunk_size

    embed_size = args.embed_size
    batch_size = 16
    num_epochs = 30
    learning_rate = 1e-3

    block_size = args.block_size
    train_subset_size = 20000
    log_interval_steps = 100
    sample_interval_seconds = 30

    max_steps_per_epoch = args.max_steps_per_epoch
    num_inner_layers = args.num_inner_mlp_layers

    # NEW: pick device from args.device_id, fallback to cpu if needed
    requested_device_id = args.device_id
    if requested_device_id.startswith("cuda") and not torch.cuda.is_available():
        print(f"Requested device '{requested_device_id}' but CUDA not available. Falling back to CPU.")
        device = torch.device("cpu")
    else:
        device = torch.device(requested_device_id)

    print(f"Using device: {device}, block_size={block_size}, kgram_k={k}, chunk_size={chunk_size}, embed_size={embed_size}")

    ############################################################################
    # Data
    ############################################################################
    combined_dataset, enc, vocab_size = load_and_prepare_data(
        args=args,
        block_size=block_size,
        train_subset_size=train_subset_size
    )

    train_loader = torch.utils.data.DataLoader(
        combined_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=seq_collate_fn
    )

    ############################################################################
    # Models
    ############################################################################
    kgram_model = KGramMLPSeqModel(
        vocab_size=vocab_size,
        k=k,
        embed_size=embed_size,
        num_inner_layers=num_inner_layers,
        chunk_size=chunk_size
    ).to(device)

    kgram_embedding_model = KGramEmbeddingSeqModel(
        vocab_size=vocab_size,
        k=k,
        embed_size=embed_size,
        num_inner_layers=num_inner_layers,
        chunk_size=chunk_size
    ).to(device)

    lstm_model = LSTMSeqModel(
        vocab_size=vocab_size,
        embed_size=embed_size,
        hidden_size=embed_size
    ).to(device)

    transformer = TransformerModel(
        vocab_size=vocab_size,
        d_model=embed_size,
        use_mla = True,
        use_swiglu = True
    ).to(device)

    models = {
    #   "kgram_mlp_seq": kgram_model,
    #   "kgram_embedding_seq": kgram_embedding_model,
    #   "lstm_seq": lstm_model,
      "kvcache_transformer": transformer,
    }


    ############################################################################
    # Train each model
    ############################################################################
    for model_name, model in models.items():
        print(f"\n=== Training model: {model_name} ===")
        train_one_model(
            model=model,
            loader=train_loader,
            epochs=num_epochs,
            model_name=model_name,
            device=device,
            lr=learning_rate,
            log_steps=log_interval_steps,
            sample_interval=sample_interval_seconds,
            max_steps_per_epoch=max_steps_per_epoch,
            enc=enc,
            prompt=args.prompt  # <--- Pass the user-specified prompt here
        )

        # Final generation from the user-provided prompt (args.prompt).
        with torch.no_grad():
            # 1) Greedy
            text_greedy, ann_greedy = generate_text(
                model, enc, args.prompt, max_new_tokens=40, device=device,
                top_p=None,
            )
            # 2) top-p=0.95
            text_topp, ann_topp = generate_text(
                model, enc, args.prompt, max_new_tokens=40, device=device,
                top_p=0.95,
            )
            # 3) top-p=1.0 => full distribution random sampling
            text_topp1, ann_topp1 = generate_text(
                model, enc, args.prompt, max_new_tokens=40, device=device,
                top_p=1.0,
            )

        print(f"[{model_name}] Final sample (greedy) from prompt: '{args.prompt}'")
        print(text_greedy)
        print(f"Annotated:\n{ann_greedy}\n")

        print(f"[{model_name}] Final sample (top-p=0.95) from prompt: '{args.prompt}'")
        print(text_topp)
        print(f"Annotated:\n{ann_topp}\n")

        print(f"[{model_name}] Final sample (top-p=1.0) from prompt: '{args.prompt}'")
        print(text_topp1)
        print(f"Annotated:\n{ann_topp1}")
        print("--------------------------------------------------")

    # Finally, let's share how I'm feeling:
    print("\n*** I'm feeling great today! Hope you're well, too. ***")

from training.generation import nucleus_sampling

from collections import Counter

def test_nucleus_sampling():
    # Example logits vector (you can replace these with your own values)
    logits = torch.tensor([2.0,3.0,4.0,4.4,5.0,5.2])
    threshold = 0.95
    
    # Original probabilities for reference
    original_probs = torch.softmax(logits, dim=-1)
    
    print("Original logits:", logits)
    print("Original probabilities:", original_probs)
    
    # Sample 100 times using nucleus_sampling
    samples = []
    for _ in range(1000):
        sampled_index = nucleus_sampling(logits, threshold)
        # Ensure that the sampled index is a Python integer
        if isinstance(sampled_index, torch.Tensor):
            sampled_index = sampled_index.item()
        samples.append(sampled_index)
    
    # Count frequency of each index
    frequencies = Counter(samples)
    
    print("Sample frequencies over 100 samples:")
    for idx in sorted(frequencies):
        print(f"Index {idx}: {frequencies[idx]}")
   
def test_overfitting_study():
    args = parse_args()
    
    # Model configurations to test
    configs = [
        {
            'name': 'tiny_transformer',
            'd_model': 256,
            'n_heads': 4,
            'n_blocks': 3,
            'use_mla': True,
            'use_swiglu': True
        },
        {
            'name': 'medium_transformer',
            'd_model': 512,
            'n_heads': 8,
            'n_blocks': 6,
            'use_mla': True,
            'use_swiglu': True
        },
        {
            'name': 'large_transformer',
            'd_model': 1024,
            'n_heads': 8,
            'n_blocks': 6,
            'use_mla': False,
            'use_swiglu': True
        }
    ]
    
    # Training parameters
    batch_size = 16
    num_epochs = 30
    block_size = args.block_size
    train_subset_size = 20000
    
    # Device setup
    device = torch.device(args.device_id if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load data using the common function
    combined_dataset, enc, vocab_size = load_and_prepare_data(
        args=args,
        block_size=block_size,
        train_subset_size=train_subset_size
    )
    
    # Split into train/test
    train_dataset, test_dataset = combined_dataset.split(test_ratio=0.2)
    
    # Create dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=seq_collate_fn
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=seq_collate_fn
    )

    results = []
    all_losses = {}
    generated_texts = {}
    for config in configs:
        print(f"\n=== Testing {config['name']} ===")
        
        # Create model
        model = TransformerModel(
            vocab_size=vocab_size,
            d_model=config['d_model'],
            n_heads=config['n_heads'],
            n_blocks=config['n_blocks'],
            use_mla=config['use_mla'],
            use_swiglu=config['use_swiglu']
        ).to(device)
        
        # Train with validation
        train_losses, test_losses = train_one_model(
            model=model,
            loader=train_loader,
            test_loader=test_loader,
            epochs=num_epochs,
            model_name=config['name'],
            device=device,
            lr=1e-3,
            log_steps=100,
            sample_interval=30,
            max_steps_per_epoch=args.max_steps_per_epoch,
            enc=enc,
            prompt=args.prompt
        )

        all_losses[config['name']] = {
            'train': train_losses,
            'test': test_losses
        }
        
        # Generate text samples with different sampling strategies
        model.eval()
        generated_samples = []
        with torch.no_grad():
            for top_p, strategy_name in [(0.95, "nucleus")]:
            # for top_p, strategy_name in [(None, "greedy"), (0.95, "nucleus"), (1.0, "random")]:
                text, ann = generate_text(
                    model, enc, args.prompt, max_new_tokens=40, device=device,
                    top_p=top_p
                )
                generated_samples.append({
                    'strategy': strategy_name,
                    'text': text,
                    'annotated': ann
                })
        
        generated_texts[config['name']] = generated_samples
        
        # Store results
        results.append({
            'config': config,
            'final_train_loss': train_losses[-1],
            'final_test_loss': test_losses[-1],
            'overfitting_gap': test_losses[-1] - train_losses[-1]
        })
    
    # Print summary of results
    print("\n=== Overfitting Study Results ===")
    print("\nModel Configurations and Their Performance:")
    for result in results:
        config = result['config']
        print(f"\nModel: {config['name']}")
        print(f"Architecture: {config['d_model']} dims, {config['n_heads']} heads, {config['n_blocks']} blocks")
        print(f"Final Train Loss: {result['final_train_loss']:.4f}")
        print(f"Final Test Loss: {result['final_test_loss']:.4f}")
        print(f"Overfitting Gap: {result['overfitting_gap']:.4f}")

        for sample in generated_texts[config['name']]:
            print(f"{sample['strategy']} annotated sampling: {sample['annotated']}")
        print("\n" + "="*80)

    try:        
        plt.style.use('ggplot')
        
        # Create a figure with subplots (one for each model)
        num_models = len(all_losses)
        fig, axes = plt.subplots(1, num_models, figsize=(6.5*num_models, 6))
        
        # Colors for consistency
        train_color = '#2E86C1'  # Blue for training
        test_color = '#E74C3C'   # Red for testing
        
        # Plot each model's results
        for idx, (model_name, losses) in enumerate(all_losses.items()):
            # Get model config for title
            model_config = next(result['config'] for result in results if result['config']['name'] == model_name)
            
            # Create subplot title with model details
            title = f"{model_name}\n({model_config['d_model']}d, {model_config['n_heads']} heads, {model_config['n_blocks']} blocks)"
            
            # Plot train and test losses
            axes[idx].plot(losses['train'], label='Train', color=train_color)
            axes[idx].plot(losses['test'], label='Test', color=test_color)
            axes[idx].set_title(title, fontsize=12)
            axes[idx].set_xlabel('Epoch')
            axes[idx].set_ylabel('Loss')
            axes[idx].legend()
            axes[idx].grid(True)
        
        # Add overall title and show
        plt.suptitle('Training vs Testing Loss for Different Model Sizes', fontsize=14)
        plt.tight_layout()
        plt.show()
        
    except ImportError:
        print("matplotlib not available - skipping loss curve plotting")


if __name__ == "__main__":
    #test_nucleus_sampling()
    # test_overfitting_study()
    main()
