from dataclasses import dataclass, field as dc_field
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from scaling_llms.utils.config import BaseJsonConfig


# -------------------------
# MODEL CONFIGS
# -------------------------
MLP_TYPE2CLASS_ACTIVATION = {
    "standard_gelu": ("mlp", "gelu"),
    "standard_silu": ("mlp", "silu"),
    "geglu": ("gated_mlp", "gelu"),
    "swiglu": ("gated_mlp", "silu"),
}
NORMALIZATION_TYPES = {"layernorm", "rmsnorm"}
POSITIONAL_ENCODING_TYPES = {"absolute", "rotary"}


@dataclass
class GPTConfig(BaseJsonConfig):
    seq_len: int = 512
    vocab_size: int = 256
    n_embd: int = 256
    n_layer: int = 4
    n_head: int = 4
    
    # MLP 
    mlp_type: str = "standard_gelu"
    mlp_bias: bool = True

    # Hidden dims
    d_ff: int | None = None  # Standard MLP (default 4*n_embd if None)
    d_ff_gated: int | None = None  # Gated MLP (parameter-matched default if None)

    # Normalization
    norm_type: str = "layernorm"
    norm_eps: float = 1e-5

    # Positional Encoding
    pos_encoding_type: str = "absolute"  # (or "rotary")
    rope_theta: float = 10000.0  # Only used if pos_encoding_type == "rotary"

    # Dropout
    attn_pdrop: float = 0.0  # On attention weights after softmax (regularizes attention routing)
    embd_pdrop: float = 0.0   # On input embeddings after token + positional embedding sum
    resid_pdrop: float = 0.0  # On sublayer outputs before adding back to residual stream (attn + MLP)
    
    # Attention
    attn_bias: bool = False

    # LM Head
    tied_embeddings: bool = True
    lm_head_bias: bool = False

    # Derived (not user inputs)
    d_head: int = dc_field(init=False)
    mlp_class: str = dc_field(init=False)
    activation: str = dc_field(init=False)


    # --- PRIVATE METHODS ---
    def __post_init__(self):
        # Head dimension
        assert self.n_embd % self.n_head == 0, "n_embd must be divisible by n_head"
        self.d_head = self.n_embd // self.n_head

        # MLP and Activation
        mlp_type = self.mlp_type.lower()
        assert mlp_type in MLP_TYPE2CLASS_ACTIVATION, (
            f"Invalid mlp_type: {self.mlp_type}"
        )
        self.mlp_type = mlp_type
        self.mlp_class, self.activation = MLP_TYPE2CLASS_ACTIVATION[self.mlp_type]

        # MLP expansion
        ## Standard
        if self.mlp_class == "mlp":
            if self.d_ff is None:
                self.d_ff = 4 * self.n_embd
            else:
                assert self.d_ff > self.n_embd, "d_ff should be larger than n_embd"

        ## Gated
        if self.mlp_class == "gated_mlp":
            if self.d_ff_gated is None:
                # Parameter-matched gated FFN width relative to standard FFN with d_ff = 4 * n_embd:
                # standard params ~ 2 * n_embd * (4 * n_embd)
                # gated params    ~ 3 * n_embd * d_ff_gated
                self.d_ff_gated = (8 * self.n_embd) // 3
            else:
                assert self.d_ff_gated > self.n_embd, "d_ff_gated should be larger than n_embd"

        # Normalization
        norm_type = self.norm_type.lower()
        assert norm_type in NORMALIZATION_TYPES, f"Invalid norm type: {self.norm_type}"
        self.norm_type = norm_type

        # Positional Encoding
        pos_encoding_type = self.pos_encoding_type.lower()
        assert pos_encoding_type in POSITIONAL_ENCODING_TYPES, (
            f"Invalid pos_encoding_type: {self.pos_encoding_type}"
        )
        self.pos_encoding_type = pos_encoding_type

        if self.pos_encoding_type == "rotary":
            assert self.rope_theta > 0, "rope_theta must be positive for rotary embeddings"
            assert self.d_head % 2 == 0, "Rotary embeddings require d_head to be even"

        # Dropout
        for name, p in {
            "attn_pdrop": self.attn_pdrop,
            "embd_pdrop": self.embd_pdrop,
            "resid_pdrop": self.resid_pdrop,
        }.items():
            assert 0.0 <= p < 1.0, f"{name} must be in [0, 1)"
            
def make_activation(name: str) -> nn.Module:
    name = name.lower()
    if name == "gelu":
        return nn.GELU(approximate="tanh")
    if name == "silu":
        return nn.SiLU()
    raise ValueError(f"Unsupported activation: {name}")


# -------------------------
# NORMALIZATION
# -------------------------
class RMSNorm(nn.Module):
    """RMSNorm with a learned scale parameter and no bias."""

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return x * rms * self.weight


def make_norm(cfg: GPTConfig) -> nn.Module:
    if cfg.norm_type == "layernorm":
        return nn.LayerNorm(cfg.n_embd, eps=cfg.norm_eps)
    if cfg.norm_type == "rmsnorm":
        return RMSNorm(cfg.n_embd, eps=cfg.norm_eps)
    raise ValueError(f"Unsupported norm type: {cfg.norm_type}")


# -------------------------
# POSITIONAL ENCODINGS
# -------------------------
def rotate_half(x: torch.Tensor) -> torch.Tensor:
    # x: (..., d_head)
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    x_rot = torch.stack((-x2, x1), dim=-1)
    return x_rot.flatten(start_dim=-2)


class RotaryEmbedding(nn.Module):
    """
    Precomputes cos/sin tables for RoPE.
    Applies to tensors of shape (B, n_head, T, d_head).
    """
    def __init__(self, dim: int, max_seq_len: int, base: float = 10000.0):
        super().__init__()
        assert dim % 2 == 0, "RoPE dim must be even"

        inv_freq = 1.0 / (
            base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        t = torch.arange(max_seq_len, dtype=torch.float32)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.repeat_interleave(freqs, repeats=2, dim=-1)

        self.register_buffer("cos_cached", emb.cos()[None, None, :, :], persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :], persistent=False)

    def forward(self, q: torch.Tensor, k: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        _, _, T, _ = q.shape
        cos = self.cos_cached[:, :, :T, :].to(dtype=q.dtype, device=q.device)
        sin = self.sin_cached[:, :, :T, :].to(dtype=q.dtype, device=q.device)
        q = (q * cos) + (rotate_half(q) * sin)
        k = (k * cos) + (rotate_half(k) * sin)
        return q, k


# -------------------------
# ATTENTION
# -------------------------
class CausalSelfAttention(nn.Module):
    
    def __init__(self, cfg):
        super().__init__()
        
        # Key variables
        self.n_head = cfg.n_head
        self.n_embd = cfg.n_embd
        self.d_head = cfg.d_head
        self.seq_len = cfg.seq_len
        self.pos_encoding_type = cfg.pos_encoding_type
        self.attn_bias = cfg.attn_bias

        # 1. Fused QKV Projection
        self.c_attn = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=self.attn_bias)

        # 2. Output Projection
        self.c_proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=self.attn_bias)

        # 3. Regularization
        self.attn_dropout = nn.Dropout(cfg.attn_pdrop)
        self.resid_dropout = nn.Dropout(cfg.resid_pdrop)

        # 4. Register Mask as a Buffer 
        # This saves you from passing 'mask' in forward() every time
        self.register_buffer(
            "causal_mask", 
            torch.tril(
                torch.ones(self.seq_len, self.seq_len, dtype=torch.bool)
            ).view(1, 1, self.seq_len, self.seq_len)
        )

        # 5. Positional Encoding (if using RoPE)
        if self.pos_encoding_type == "rotary":
            self.rope = RotaryEmbedding(
                dim=self.d_head,
                max_seq_len=cfg.seq_len,
                base=cfg.rope_theta,
            )
        else:
            self.rope = None

    def forward(self, x):
        B, T, D = x.shape # (Batch, Seq_Len, Embedding/Model_Dim)

        # --- STEP 1: Calculate Q, K, V ---
        # Run the linear layer
        # Shape: (B, T, 3 * D)
        qkv = self.c_attn(x) 

        # Reshape to isolate heads and QKV 
        # We move the '3' to the front (dim 0) for easy unpacking
        # Shape: (3, B, n_head, T, d_head)
        qkv = qkv.view(B, T, 3, self.n_head, self.d_head).permute(2, 0, 3, 1, 4)

        # Unpack: Results are now cleanly (B, n_head, T, d_head)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Optional RoPE application
        if self.rope is not None:
            q, k = self.rope(q, k)

        # --- STEP 2: Attention Mechanism ---
        # (B, n_head, T, d_head) @ (B, n_head, d_head, T) -> (B, n_head, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.d_head))
        
        # Apply Causal Mask (using the registered buffer)
        # NOTE: Slice the mask to T because the current batch may be shorter than cfg.seq_len
        att = att.masked_fill(~self.causal_mask[:,:,:T,:T], float('-inf'))
        
        # Softmax & Dropout
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        
        # Weighted Sum
        # (B, n_head, T, T) @ (B, n_head, T, d_head) -> (B, n_head, T, d_head)
        y = att @ v

        # --- STEP 3: Reassemble ---
        # 1. Transpose: Swap Head and Time back -> (B, T, n_head, d_head)
        # 2. Contiguous: Fix memory layout
        # 3. View: Merge H and d -> (B, T, D)
        y = y.transpose(1, 2).contiguous().view(B, T, D)

        # --- STEP 4: Output Projection ---
        y = self.resid_dropout(self.c_proj(y))
        
        return y
    

# -------------------------
# MLPs
# -------------------------
class StandardMLP(nn.Module):

    def __init__(self, cfg):
        super().__init__()

        # 1. Expansion: n_embd -> d_ff > n_embd
        self.c_fc = nn.Linear(cfg.n_embd, cfg.d_ff, bias=cfg.mlp_bias)

        # 2. Activation
        self.act = make_activation(cfg.activation)

        # 3. Projection: d_ff -> n_embd
        self.c_proj = nn.Linear(cfg.d_ff, cfg.n_embd, bias=cfg.mlp_bias)

        # 4. Regularization
        self.dropout = nn.Dropout(cfg.resid_pdrop)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.act(x)
        x = self.c_proj(x)
        x = self.dropout(x)

        return x

class GatedMLP(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.c_gate = nn.Linear(cfg.n_embd, cfg.d_ff_gated, bias=cfg.mlp_bias)
        self.c_val = nn.Linear(cfg.n_embd, cfg.d_ff_gated, bias=cfg.mlp_bias)
        self.act = make_activation(cfg.activation)
        self.c_proj = nn.Linear(cfg.d_ff_gated, cfg.n_embd, bias=cfg.mlp_bias)
        self.dropout = nn.Dropout(cfg.resid_pdrop)

    def forward(self, x):
        gate = self.act(self.c_gate(x))
        val = self.c_val(x)
        x = gate * val
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


def make_mlp(cfg: GPTConfig) -> nn.Module:
    if cfg.mlp_class == "mlp":
        return StandardMLP(cfg)
    elif cfg.mlp_class == "gated_mlp":
        return GatedMLP(cfg)
    else:
        raise ValueError(f"Unsupported mlp_class: {cfg.mlp_class}")


# -------------------------
# TRANSFORMER BLOCK
# -------------------------
class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        
        # 1. Norms (Pre-Norm architecture)
        self.norm1 = make_norm(cfg)
        self.norm2 = make_norm(cfg)
        
        # 2. Sub-layers
        self.attn = CausalSelfAttention(cfg)
        self.mlp = make_mlp(cfg)
       
    def forward(self, x):
        # Weighted sum (Communication)
        # Note the Pre-Norm placement: attn(norm1(x))
        x = x + self.attn(self.norm1(x))
        
        # Element-wise processing (Computation)
        x = x + self.mlp(self.norm2(x))
        
        return x
    

# -------------------------
# GPT MODEL
# -------------------------
@dataclass(slots=True)
class GPTOutput:
    logits: torch.Tensor
    loss: torch.Tensor | None

class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        # A. The Transformer
        transformer_dict = dict()

        ## A.1. Embeddings
        transformer_dict["wte"] = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        if cfg.pos_encoding_type == "absolute":
            transformer_dict["wpe"] = nn.Embedding(cfg.seq_len, cfg.n_embd)

        ## A.2. The Stack (Dropout usually applied after embedding sum)
        transformer_dict["embd_drop"] = nn.Dropout(cfg.embd_pdrop)

        ## A.3. The Layers (ModuleList allows standard iteration)
        transformer_dict["h"] = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])

        ## A.4. Final Norm
        transformer_dict["norm_f"] = make_norm(cfg)

        self.transformer = nn.ModuleDict(transformer_dict)
    
        # B. The Language Modeling Head
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=cfg.lm_head_bias)

        # C. Init weights
        self.apply(self._init_weights)

        # D. Weight Tying (Optional)
        if cfg.tied_embeddings:
            self.lm_head.weight = self.transformer.wte.weight

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None, loss_reduction="mean") -> GPTOutput:
        B, T = idx.size()

        assert T <= self.cfg.seq_len, (
            f"Sequence length {T} exceeds configured seq_len={self.cfg.seq_len}"
        )

        # 1. Embeddings
        x = self.transformer.wte(idx)  # Token Emb (B, T, D)

        if self.cfg.pos_encoding_type == "absolute":
            # Create Position Indices (0, 1, ..., T-1)
            pos = torch.arange(0, T, dtype=torch.long, device=idx.device)

            # Add Positional Embeddings of shape (T, D)
            x = x + self.transformer.wpe(pos)

        # 2. Dropout after embeddings
        x = self.transformer.embd_drop(x)

        # 3. Blocks
        for block in self.transformer.h:
            x = block(x)

        # 4. Final Norm
        x = self.transformer.norm_f(x)

        # 5. Output Head
        if targets is not None:
            # If we are training, we only need the logits for the last few tokens
            # or all tokens depending on implementation. Here we compute all.
            logits = self.lm_head(x)

            # Flatten for CrossEntropy: (B*T, Vocab)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
                reduction=loss_reduction
            )

        else:
            # Inference optimization: usually we only care about the LAST token's logits
            # But for general correctness, we return all.
            logits = self.lm_head(x[:, [-1], :]) # Shape (B, 1, Vocab)
            loss = None

        return GPTOutput(logits=logits, loss=loss)