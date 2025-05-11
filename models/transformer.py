import math
import torch
import torch.nn as nn
import torch.nn.functional as F

ROPE = True

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization"""
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
        return x / norm * self.weight

class MultiHeadAttention(nn.Module):
    """Standard Multi-Head Attention"""
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        
        self.qkv_proj = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        B, T, C = x.shape
        qkv = self.qkv_proj(x).split(self.d_model, dim=2)
        
        # Split into heads
        q, k, v = [y.view(B, T, self.n_heads, self.head_dim).transpose(1, 2) 
                   for y in qkv]
        
        # Attention scores
        attn = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
        if mask is not None:
            attn = attn.masked_fill(mask == 0, float('-inf'))
        attn = F.softmax(attn, dim=-1)
        
        # Combine heads
        y = (attn @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(y)

class MultiHeadLatentAttention(nn.Module):
    """DeepSeek-V2's Multi-head Latent Attention with RoPE support"""
    def __init__(self, d_model: int, n_heads: int, rope_theta: float = 10000.0):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.kv_compression_ratio = 1.0  # From DeepSeek-V2 paper = 0.25
        
        # Key-Value compression params (d_c = 4d_h per paper)
        self.kv_compressed_dim = int(4 * self.head_dim)
        
        # Projections
        self.q_proj = nn.Linear(d_model, d_model)
        self.kv_compressor = nn.Linear(d_model, self.kv_compressed_dim)
        self.k_up_proj = nn.Linear(self.kv_compressed_dim, d_model)
        self.v_up_proj = nn.Linear(self.kv_compressed_dim, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        # RoPE
        self.rope = RotaryEmbedding(
            dim=self.head_dim,
            theta=rope_theta,
            decoupled=True  # Special MLA-style RoPE
        )
                
    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        B, T, C = x.shape
        
        # Compress keys and values
        kv_compressed = self.kv_compressor(x)  # (B, T, compression_dim)
        k = self.k_up_proj(kv_compressed)      # (B, T, d_model)
        v = self.v_up_proj(kv_compressed)      # (B, T, d_model)
        
        # Project queries
        q = self.q_proj(x)                     # (B, T, d_model)
        
        # Split into heads
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        if ROPE:
          # Apply decoupled RoPE (different for q and k)
          q = self.rope(q, offset=0)  # Full RoPE for queries
          k = self.rope(k, offset=0, apply_rotary=False)  # Only position IDs for keys
          
          # Attention scores
          attn = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
          if mask is not None:
              attn = attn.masked_fill(mask == 0, float('-inf'))
          attn = F.softmax(attn, dim=-1)
          
          # Combine heads
          y = (attn @ v).transpose(1, 2).reshape(B, T, self.d_model)
          return self.out_proj(y)
        else:
          # Attention scores
          attn = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
          if mask is not None:
              attn = attn.masked_fill(mask == 0, float('-inf'))
          attn = F.softmax(attn, dim=-1)
          
          # Combine heads
          y = (attn @ v).transpose(1, 2).contiguous().view(B, T, C)
          return self.out_proj(y)


class RotaryEmbedding(nn.Module):
    """Modified RoPE for MLA's decoupled strategy"""
    def __init__(self, dim: int, theta: float = 10000.0, decoupled: bool = True):
        super().__init__()
        self.dim = dim
        self.theta = theta
        self.decoupled = decoupled
        
        # Standard RoPE frequencies
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        
        # MLA-specific decoupling
        if decoupled:
            self.k_rope = nn.Linear(dim, dim, bias=False)
        
    def forward(self, x: torch.Tensor, offset: int, apply_rotary: bool = True):
        seq_len = x.shape[-2]
        t = torch.arange(offset, offset + seq_len, device=x.device).type_as(self.inv_freq)

        if self.decoupled and not apply_rotary:
            # MLA's position-only projection for keys
            t = t.view(1,1,-1,1)
            return self.k_rope(x * t)
        
        # Standard RoPE implementation
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1).to(x.device)
        return x * emb.cos() + self.rotate_half(x) * emb.sin()
    
    def rotate_half(self, x):
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)

class FeedForward(nn.Module):
    """Position-wise Feed Forward Network"""
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.SiLU(),  # SwiGLU activation could also be used here
            nn.Linear(d_ff, d_model)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class SwiGLUFFN(nn.Module):
    """SwiGLU activation function"""
    def __init__(self, dim: int, hidden_dim: int = None, multiple_of: int = 256):
        super().__init__()
        hidden_dim = int(2*dim*4/3) if hidden_dim is None else hidden_dim
        # ensure hidden_dim is a multiple of multiple_of (improves hardward utilization per llama)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
        self.hidden_dim = hidden_dim
        self.w1 = nn.Linear(dim, hidden_dim, bias=False) # gate proj
        self.w2 = nn.Linear(dim, hidden_dim, bias=False) # value proj
        self.w3 = nn.Linear(hidden_dim, dim, bias=False) # output proj

        self.activation = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w3(self.activation(self.w1(x)) * self.w2(x))

class TransformerBlock(nn.Module):
    """Single Transformer Block"""
    def __init__(self, d_model: int, n_heads: int, d_ff: int, 
                 use_mla: bool = True, use_swiglu: bool = True, norm_eps: float = 1e-5):
        super().__init__()
        self.norm1 = RMSNorm(d_model, eps=norm_eps)
        self.attn = MultiHeadLatentAttention(d_model, n_heads) if use_mla else MultiHeadAttention(d_model, n_heads)
        self.norm2 = RMSNorm(d_model, eps=norm_eps)
        self.ffn = SwiGLUFFN(d_model) if use_swiglu else FeedForward(d_model, d_ff)
        
    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        # Self-attention with skip connection
        x = x + self.attn(self.norm1(x), mask)
        # Feed-forward with skip connection
        x = x + self.ffn(self.norm2(x))
        return x

class TransformerModel(nn.Module):
    """Decoder-only Transformer Model"""
    def __init__(self, vocab_size: int = 50257, d_model: int = 1024, 
                 n_heads: int = 8, n_blocks: int = 6, d_ff: int = 4096,
                 use_mla: bool = False, use_swiglu: bool = True, norm_eps: float = 1e-5):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Parameter(torch.zeros(1, 2048, d_model))  # Max sequence length
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, use_mla, use_swiglu, norm_eps)
            for _ in range(n_blocks)
        ])
        
        # Final normalization and output projection
        self.ln_f = RMSNorm(d_model, eps=norm_eps)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif hasattr(module, 'pos_emb'):
            nn.init.normal_(module.pos_emb, mean=0.0, std=0.02)
            
    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        B, T = idx.shape

        # print(f"Input shape: {idx.shape}")
        # print(f"Sample tokens: {idx[0,:5] if T > 5 else idx[0]}")
        assert T > 1, "Sequence length must be greater than 1"
        
        # Get token and position embeddings
        tok_emb = self.token_emb(idx)  # (B, T, d_model)
        pos_emb = self.pos_emb[:, :T, :]  # (1, T, d_model)
        x = tok_emb + pos_emb
        
        # Create causal mask
        # mask = torch.tril(torch.ones(T, T)).view(1, 1, T, T).to(idx.device)
        # print(f"Mask shape: {mask.shape}")
        # print(f"Mask: {mask[0,0]}")
        
        mask = torch.tril(torch.ones(T, T, device=idx.device)).view(1, 1, T, T)
        # print(f"new Mask shape: {mask.shape}")
        # print(f"new Mask: {mask[0,0]}")

        # Apply transformer blocks
        for block in self.blocks:
            x = block(x, mask)
            
        # Final projection
        x = self.ln_f(x)
        logits = self.head(x)
        return logits


'''
### Comparison Table:

| Feature               | Standard Attention | MLA (DeepSeek-V2) |
|-----------------------|--------------------|-------------------|
| KV Cache Size         | O(n_layers × seq_len × dim) | O(n_layers × seq_len × dim/4) |
| Position Handling     | Full RoPE for Q/K  | Decoupled RoPE    |
| Memory Efficiency     | Lower              | 93% reduction     |
| Performance           | Baseline           | Comparable/better |
'''
