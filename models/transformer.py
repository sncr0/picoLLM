import math
import torch
import torch.nn as nn
import torch.nn.functional as F

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
    """DeepSeek-V2's Multi-head Latent Attention"""
    def __init__(self, d_model: int, n_heads: int, kv_compression_ratio: float = 0.25):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.kv_compression_dim = int(self.head_dim * kv_compression_ratio)
        
        # Key-Value compression
        self.kv_compressor = nn.Linear(d_model, self.kv_compression_dim)
        self.k_up_proj = nn.Linear(self.kv_compression_dim, d_model)
        self.v_up_proj = nn.Linear(self.kv_compression_dim, d_model)
        
        # Query projection
        self.q_proj = nn.Linear(d_model, d_model)
        
        # Output projection
        self.out_proj = nn.Linear(d_model, d_model)
        
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
        
        # Attention scores
        attn = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
        if mask is not None:
            attn = attn.masked_fill(mask == 0, float('-inf'))
        attn = F.softmax(attn, dim=-1)
        
        # Combine heads
        y = (attn @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(y)

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

class TransformerBlock(nn.Module):
    """Single Transformer Block"""
    def __init__(self, d_model: int, n_heads: int, d_ff: int, 
                 use_mla: bool = False, norm_eps: float = 1e-5):
        super().__init__()
        self.ln1 = RMSNorm(d_model, eps=norm_eps)
        self.attn = MultiHeadLatentAttention(d_model, n_heads) if use_mla else MultiHeadAttention(d_model, n_heads)
        self.ln2 = RMSNorm(d_model, eps=norm_eps)
        self.ffn = FeedForward(d_model, d_ff)
        
    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        # Self-attention with skip connection
        x = x + self.attn(self.ln1(x), mask)
        # Feed-forward with skip connection
        x = x + self.ffn(self.ln2(x))
        return x

class TransformerModel(nn.Module):
    """Decoder-only Transformer Model"""
    def __init__(self, vocab_size: int = 50257, d_model: int = 1024, 
                 n_heads: int = 8, n_blocks: int = 6, d_ff: int = 4096,
                 use_mla: bool = False, norm_eps: float = 1e-5):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Parameter(torch.zeros(1, 2048, d_model))  # Max sequence length
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, use_mla, norm_eps)
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
            
    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        B, T = idx.shape
        
        # Get token and position embeddings
        tok_emb = self.token_emb(idx)  # (B, T, d_model)
        pos_emb = self.pos_emb[:, :T, :]  # (1, T, d_model)
        x = tok_emb + pos_emb
        
        # Create causal mask
        mask = torch.tril(torch.ones(T, T)).view(1, 1, T, T).to(idx.device)
        
        # Apply transformer blocks
        for block in self.blocks:
            x = block(x, mask)
            
        # Final projection
        x = self.ln_f(x)
        logits = self.head(x)
        return logits

# class TransformerModel(nn.Module):
#     def __init__(self, vocab_size=50257, d_model=1024, n_heads=2, n_blocks=4):
#         super().__init__()
#         pass
