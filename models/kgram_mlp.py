import torch
import torch.nn as nn
import torch.nn.functional as F

# first approach: one-hot embedding + MLP
class KGramMLPSeqModel(nn.Module):
    """
    For each position t in [0..seq_len-1], gather the last k tokens => one-hot => MLP => logits.
    Return (seq_len, batch, vocab_size).

    Potentially very large memory usage for big vocab or seq_len. chunk_size helps mitigate overhead.
    """

    def __init__(self, vocab_size, k=3, embed_size=1024, num_inner_layers=1, chunk_size=1):
        super().__init__()
        self.k = k
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.num_inner_layers = num_inner_layers
        self.chunk_size = chunk_size

        # one-hot embedding
        #     simple to implement, preserves exact token counts
        #     but requires more memory (O(k*vocab_size) input dim)
        #     and more compute
        input_size = k * vocab_size

        layers = []
        current_size = input_size
        for _ in range(num_inner_layers):
            layers.append(nn.Linear(current_size, embed_size))
            layers.append(nn.SiLU())
            current_size = embed_size
        layers.append(nn.Linear(current_size, vocab_size))
        self.net = nn.Sequential(*layers)

    def forward(self, tokens_seq):
        """
        tokens_seq: (seq_len, batch)
        return: (seq_len, batch, vocab_size)
        We'll do a loop over time steps. chunk_size can reduce overhead.
        """
        seq_len, batch_size = tokens_seq.shape
        outputs = []

        start = 0
        while start < seq_len:
            end = min(start + self.chunk_size, seq_len)
            block_outputs = []
            for t in range(start, end):
                batch_logits = []
                for b in range(batch_size):
                    if t < self.k:
                        needed = self.k - t
                        context_ids = [0]*needed + tokens_seq[:t, b].tolist()
                    else:
                        context_ids = tokens_seq[t-self.k:t, b].tolist()

                    context_oh = F.one_hot(
                        torch.tensor(context_ids, dtype=torch.long, device=tokens_seq.device),
                        num_classes=self.vocab_size
                    )
                    context_flat = context_oh.flatten().float().unsqueeze(0)
                    logits_b = self.net(context_flat)  # (1, vocab_size)
                    batch_logits.append(logits_b)
                block_outputs.append(torch.cat(batch_logits, dim=0).unsqueeze(0))  # (1, batch, vocab_size)

            block_outputs = torch.cat(block_outputs, dim=0)  # (chunk_size, batch, vocab_size)
            outputs.append(block_outputs)
            start = end

        outputs = torch.cat(outputs, dim=0)  # (seq_len, batch, vocab_size)
        return outputs


# alternate approach using embeddings:
#     more memory/compute efficient, learns token representations
class KGramEmbeddingSeqModel(nn.Module):
    """
    A sequence-to-sequence model that processes text using k-gram embeddings followed by an MLP.
    """
    def __init__(self, vocab_size, k=3, embed_size=1024, num_inner_layers=1, chunk_size=1):
        super().__init__()
        self.k = k
        self.vocab_size = vocab_size
        self.embed_size = (embed_size // k ) * k  # round down to nearest multiple of k (caused problems before)
        self.num_inner_layers = num_inner_layers
        self.chunk_size = chunk_size

        # Embedding layer instead of one-hot
        self.embed = nn.Embedding(vocab_size, embed_size//k)
        # input_size = embed_size  # k * (embed_size//k)
        
        # Rest of the network remains similar
        layers = []
        # current_size = input_size
        
        for _ in range(num_inner_layers):
            layers.append(nn.Linear(self.embed_size, self.embed_size))
            layers.append(nn.SiLU())
            
        layers.append(nn.Linear(self.embed_size, vocab_size))
        self.net = nn.Sequential(*layers)

    def forward(self, tokens_seq):
        seq_len, batch_size = tokens_seq.shape
        outputs = []
        
        for t in range(seq_len):
            # Get k-gram window
            if t < self.k:
                needed = self.k - t
                context = torch.cat([
                    torch.zeros(needed, batch_size, dtype=torch.long, device=tokens_seq.device),
                    tokens_seq[:t]
                ], dim=0)
            else:
                context = tokens_seq[t-self.k:t]
                
            # Embed and flatten
            embedded = self.embed(context)  # (k, batch, embed_size//k)
            embedded = embedded.view(batch_size, -1)  # (batch, k*(embed_size//k))
            
            logits = self.net(embedded).unsqueeze(0)  # (1, batch, vocab_size)
            outputs.append(logits)
        
        return torch.cat(outputs, dim=0)  # (seq_len, batch, vocab_size)
