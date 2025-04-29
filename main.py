import argparse
import random
import torch
import matplotlib.pyplot as plt
import os

from data import load_and_prepare_data, seq_collate_fn
from training import train_one_model, generate_text
from models import KGramMLPSeqModel, LSTMSeqModel, TransformerModel, KGramEmbeddingSeqModel
from analysis import monosemantic_analysis_for_token
from config import parse_args
from testing import test_nucleus_sampling, test_overfitting_study

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
    num_epochs = args.num_epochs
    learning_rate = 1e-3

    block_size = args.block_size
    train_subset_size = 20000
    log_interval_steps = 100
    sample_interval_seconds = 30

    max_steps_per_epoch = args.max_steps_per_epoch
    num_inner_layers = args.num_inner_mlp_layers

    save_model = args.save_model

    temperature = args.temperature
    dynamic_p = args.dynamic_p

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

        
        ##############################################################################
        # save model locally !!!
        if save_model:
            res_path = f"{os.getcwd()}/results"
            if not os.path.isdir(res_path):
                os.mkdir(path=res_path)
            save_path = f"{res_path}/{model_name}.pt"
            torch.save(model.state_dict(), save_path)
        ##############################################################################

        # Final generation from the user-provided prompt (args.prompt).
        with torch.no_grad():
            temps = {"low": 0.7, "high": 1.5}
            
            # 1) Greedy
            text_greedy, ann_greedy = generate_text(
                model, enc, args.prompt, max_new_tokens=40, device=device,
                top_p=None, temperature=1.0
            )
            text_greedy_lowT, ann_greedy_lowT = generate_text(
                model, enc, args.prompt, max_new_tokens=40, device=device,
                top_p=None, temperature=temps["low"]
            )
            text_greedy_highT, ann_greedy_highT = generate_text(
                model, enc, args.prompt, max_new_tokens=40, device=device,
                top_p=None, temperature=temps["high"]
            )

            # 2) top-p=0.95
            text_topp, ann_topp = generate_text(
                model, enc, args.prompt, max_new_tokens=40, device=device,
                top_p=0.95, temperature=1.0
            )
            text_topp_lowT, ann_topp_lowT = generate_text(
                model, enc, args.prompt, max_new_tokens=40, device=device,
                top_p=0.95, temperature=temps["low"]
            )
            text_topp_highT, ann_topp_highT = generate_text(
                model, enc, args.prompt, max_new_tokens=40, device=device,
                top_p=0.95, temperature=temps["high"]
            )

            # 3) top-p=1.0 => full distribution random sampling
            text_topp1, ann_topp1 = generate_text(
                model, enc, args.prompt, max_new_tokens=40, device=device,
                top_p=1.0, temperature=1.0
            )
            text_topp1_lowT, ann_topp1_lowT = generate_text(
                model, enc, args.prompt, max_new_tokens=40, device=device,
                top_p=1.0, temperature=temps["low"]
            )
            text_topp1_highT, ann_topp1_highT = generate_text(
                model, enc, args.prompt, max_new_tokens=40, device=device,
                top_p=1.0, temperature=temps["high"]
            )

            # 4) dynamic-p sampling
            text_dynamic, ann_dynamic = generate_text(
                model, enc, args.prompt, max_new_tokens=40, device=device,
                top_p=0.95, temperature=1.0, dynamic_p=True
            )
            text_dynamic_lowT, ann_dynamic_lowT = generate_text(
                model, enc, args.prompt, max_new_tokens=40, device=device,
                top_p=0.95, temperature=temps["low"], dynamic_p=True
            )
            text_dynamic_highT, ann_dynamic_highT = generate_text(
                model, enc, args.prompt, max_new_tokens=40, device=device,
                top_p=0.95, temperature=temps["high"], dynamic_p=True
            )

            print(f"[{model_name}] Greedy (T=1.0):\n{text_greedy}\nAnnotated:\n{ann_greedy}\n")
            print(f"[{model_name}] Greedy (T=0.7):\n{text_greedy_lowT}\n")
            print(f"[{model_name}] Greedy (T=1.5):\n{text_greedy_highT}\n")

            print(f"[{model_name}] Top-p=0.95 (T=1.0):\n{text_topp}\nAnnotated:\n{ann_topp}\n")
            print(f"[{model_name}] Top-p=0.95 (T=0.7):\n{text_topp_lowT}\n")
            print(f"[{model_name}] Top-p=0.95 (T=1.5):\n{text_topp_highT}\n")

            print(f"[{model_name}] Top-p=1.0 (T=1.0):\n{text_topp1}\nAnnotated:\n{ann_topp1}\n")
            print(f"[{model_name}] Top-p=1.0 (T=0.7):\n{text_topp1_lowT}\n")
            print(f"[{model_name}] Top-p=1.0 (T=1.5):\n{text_topp1_highT}\n")

            print(f"[{model_name}] Dynamic-p (T=1.0):\n{text_dynamic}\nAnnotated:\n{ann_dynamic}\n")
            print(f"[{model_name}] Dynamic-p (T=0.7):\n{text_dynamic_lowT}\n")
            print(f"[{model_name}] Dynamic-p (T=1.5):\n{text_dynamic_highT}\n")
        print("--------------------------------------------------")

    # Finally, let's share how I'm feeling:
    print("\n*** I'm feeling great today! Hope you're well, too. ***")


if __name__ == "__main__":
    #test_nucleus_sampling()
    # test_overfitting_study()
    main()
