# Pico-LLM

A lightweight and modular framework for building and experimenting with compact language models, including causal decoder-only transformers and k-gram MLPs. Designed to support rapid iteration on model architecture, training procedures, and text generation techniques with minimal hardware requirements.

## Features

- **Modular Transformer Implementation**  
  Customizable causal decoder-only transformer with configurable number of blocks, attention heads, normalization strategies (LayerNorm, RMSNorm), and embedding layers.

- **K-Gram MLP Models**  
  Sequence-to-sequence MLP model with a sliding window interface for compact input-output mappings.

- **Sampling Strategies**  
  Includes standard softmax sampling as well as nucleus (top-p) sampling for more controlled text generation.

- **Custom Data Support**  
  Easily load and train on your own sequence datasets, including non-language tokens.

- **Minimal Dependencies**  
  Runs on CPU with adjustable model sizes and memory-saving options; suitable for experimentation without specialized hardware.

## Quick Start

### Installation

```bash
git clone https://github.com/yourusername/pico-llm.git
cd pico-llm
pip install -r requirements.txt
```

### Run a Sanity Check

Run a minimal LSTM on TinyStories with memory-safe defaults:

```bash
python main.py --block size 32 --tinystories weight 0.0 --input files 3seqs.txt --prompt "0 1 2 3 4"
```

### Training with TransformerModel

To train using a causal decoder-only transformer:

```bash
python main.py --model transformer --n_layers 4 --embed_dim 128 --n_heads 4
```

### Generate Text

```bash
python generate.py --model transformer --top_p 0.9 --prompt "The quick brown"
```

## Directory Structure

```
pico-llm/
├── analysis
│   ├── __init__.py
│   └── monosemantic.py
├── config.py
├── data
│   ├── __init__.py
│   ├── collate.py
│   ├── dataset.py
│   └── tokenizer.py
├── main.py
├── models
│   ├── __init__.py
│   ├── base.py
│   ├── kgram_mlp.py
│   ├── lstm.py
│   ├── transformer.py
│   └── utils.py
├── pico_llm_proj.ipynb
└── training
    ├── __init__.py
    ├── generation.py
    └── trainer.py

```

## Notes

- Models support arbitrary vocabularies and token sets.
- Positional embeddings can be enabled or customized.
- For deeper interpretability, attention maps and intermediate activations can be visualized or logged.

## Future Work

- Integration of RoPE and NoPE positional embeddings
- Expanded interpretability tools (e.g., attention visualizations)
- Model checkpointing and resuming
- Enhanced dataset handling and augmentation

## License

MIT License
