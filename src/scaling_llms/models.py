from dataclasses import dataclass, field as dc_field
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from scaling_llms.utils.config import BaseJsonConfig


"""
Maximal Update Parameterization (μP) vs Standard Parameterization (SP)
=======================================================================

                        SP                          μP
                        ──────────────────────────────────────────────
Embedding Init. Var.    σ²_base                     σ²_base
Embedding LR            η_base                      η_base
Embedding Fwd.          x @ W_emb                   α_input · x @ W_emb
Hidden Init. Var.       σ²_base                     σ²_base / m_d
Hidden LR (Adam)        η_base                      η_base / m_d
Output Logit Fwd.       x @ W_emb.T                 α_output · x @ W_emb.T / m_d
Attention Logits        Q.T @ K / sqrt(d_head)      Q.T @ K / d_head

Where:
    m_d     : width multiplier = d / d_base (ratio of current to base width)
    d_base  : base model width (proxy model used for LR tuning)
    σ²_base : base init variance (e.g. 1/d_base)
    η_base  : base learning rate (tuned once at d_base, transferred to all widths)
    α_input : input multiplier (tunable scalar, default 1.0)
    α_output: output multiplier (tunable scalar, default 1.0)

Key properties:
    - Embeddings: init and LR unchanged from SP — no width scaling
    - Hidden weights: both init variance and LR shrink as 1/m_d
    - Output logits: scaled down by m_d in forward pass (replaces zero-init convention)
    - Attention: scale changes from 1/sqrt(d_head) to 1/d_head under μP
"""

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
    # Dimensions
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

    # Parametrization
    mup_base_width: int | None = None
    ## SP: None
    ## μP: set to the n_embd of your small proxy model.
    mup_input_mult:  float = 1.0
    mup_output_mult: float = 1.0
    ## μP forward-pass multipliers (only meaningful when using_mup=True)
    ## Default 1.0 = no-op. Expose for coord checks and ablations.

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
    
    @property
    def using_mup(self) -> bool:
        return self.mup_base_width is not None

            
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
        self.attn_pdrop = cfg.attn_pdrop

        # 1. Fused QKV Projection
        self.c_attn = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=self.attn_bias)

        # 2. Output Projection
        self.c_proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=self.attn_bias)

        # 3. Regularization
        self.resid_dropout = nn.Dropout(cfg.resid_pdrop)

        # 4. Positional Encoding (if using RoPE)
        if self.pos_encoding_type == "rotary":
            self.rope = RotaryEmbedding(
                dim=self.d_head,
                max_seq_len=cfg.seq_len,
                base=cfg.rope_theta,
            )
        else:
            self.rope = None

        # 5. μP uses 1/d_head instead of 1/sqrt(d_head).
        # (Only relevant when d_head scales with n_embd, i.e., fixed n_head.)
        # PyTorch's SDPA accepts an explicit scale kwarg (requires torch >= 2.1).
        if cfg.using_mup:
            self.attn_scale = 1.0 / cfg.d_head
        else:
            self.attn_scale = 1.0 / math.sqrt(cfg.d_head)

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
        q, k, v = qkv.unbind(0)

        # Optional RoPE application
        if self.rope is not None:
            q, k = self.rope(q, k)

        # --- STEP 2: Attention Mechanism ---
        # SDPA -> (B, n_head, T, d_head)
        y = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=self.attn_pdrop if self.training else 0.0,
            is_causal=True, # handles the causal mask for you
            scale=self.attn_scale,   # override the default
        )

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

        # E. Residual projection scaling (both SP and μP)
        ## Scale c_proj layers (attn output + MLP down-proj) by 1/sqrt(2*L)
        ## to prevent residual stream variance from growing with depth.
        for name, p in self.named_parameters():
            if name.endswith("c_proj.weight"):
                p.data.mul_((2 * cfg.n_layer) ** -0.5)

        # F. μP: zero-init the output projection (only when not tied to wte)
        if cfg.using_mup and not cfg.tied_embeddings:
            nn.init.zeros_(self.lm_head.weight)

    # --- PRIVATE METHODS ---
    def _init_weights(self, module: nn.Module) -> None:
        """
        SP init: all linears get the same std (width-independent).
        μP init w/ AdamW: hidden linears scale as 1/sqrt(n_embd); embeddings stay at base std.
        """
        d = self.cfg.n_embd

        if self.cfg.using_mup:
            d_base = self.cfg.mup_base_width

            # Embedding init: fixed at base std regardless of width
            input_std  = d_base ** -0.5
            # Hidden init: 1/sqrt(d) — same as SP at this width,
            # μP transfer comes from LR scaling, not init std change
            hidden_std = d ** -0.5
        else:
            # SP: width-dependent (fixes your existing bug from the review)
            input_std  = d ** -0.5
            hidden_std = d ** -0.5

        if isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=input_std)
        elif isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=hidden_std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        # RMSNorm / LayerNorm weights are already ones by default — leave them.

    def _encode(self, idx: torch.Tensor) -> torch.Tensor:
        """Shared: embeddings → blocks → final norm."""
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

        # α_input: scales embedding output before entering residual stream
        # No-op when mup_input_mult=1.0 or using_mup=False
        if self.cfg.using_mup and self.cfg.mup_input_mult != 1.0:
            x = x * self.cfg.mup_input_mult

        # 2. Dropout after embeddings
        x = self.transformer.embd_drop(x)

        # 3. Blocks
        for block in self.transformer.h:
            x = block(x)

        # 4. Final Norm
        x = self.transformer.norm_f(x)

        return x

    def _decode(self, x: torch.Tensor) -> torch.Tensor:
        """Shared: final norm output → logits with optional μP scaling."""
        logits = self.lm_head(x)

        if self.cfg.using_mup:
            # α_output: scales logits before loss
            # In μP the output is also divided by m_d = n_embd / mup_base_width
            # α_output lets you tune that scale independently
            m_d = self.cfg.n_embd / self.cfg.mup_base_width
            logits = logits * (self.cfg.mup_output_mult / m_d)

        return logits

    # --- API ---
    def get_param_groups(self, base_lr: float, weight_decay: float) -> list[dict]:
        """
        Returns optimizer param groups.

        In SP: all params share base_lr (with decay/no-decay split).
        In μP: hidden weights (all linears) get lr = base_lr * (d_base / d).
            Embeddings, norms, biases, and lm_head keep base_lr.

        Usage:
            groups = model.get_param_groups(base_lr=3e-4, weight_decay=0.1)
            optimizer = torch.optim.AdamW(groups)
        """
        if not self.cfg.using_mup:
            # SP: standard decay / no-decay split, same LR for all
            decay, no_decay = [], []
            for name, p in self.named_parameters():
                if not p.requires_grad:
                    continue
                if p.ndim < 2 or "bias" in name:
                    no_decay.append(p)
                else:
                    decay.append(p)
            return [
                {"params": decay,    "lr": base_lr, "weight_decay": weight_decay},
                {"params": no_decay, "lr": base_lr, "weight_decay": 0.0},
            ]

        # μP: three groups by parameter type
        d = self.cfg.n_embd
        d_base = self.cfg.mup_base_width
        hidden_lr = base_lr * (d_base / d)   # shrinks for wider models

        def _classify(name: str, p: nn.Parameter) -> str:
            # Output head (untied only — tied weights are classified as input)
            if "lm_head" in name and not self.cfg.tied_embeddings:
                return "output"
            # Embeddings, norms, biases → input LR
            if "wte" in name or "wpe" in name:
                return "input"
            if p.ndim < 2 or "bias" in name or "norm" in name:
                return "input"
            # Everything else (Q, K, V, O projections; MLP up/gate/down) → hidden LR
            return "hidden"

        groups: dict[str, dict] = {
            "input_decay":   {"params": [], "lr": base_lr,   "weight_decay": weight_decay},
            "input_nodecay": {"params": [], "lr": base_lr,   "weight_decay": 0.0},
            "hidden_decay":  {"params": [], "lr": hidden_lr, "weight_decay": weight_decay},
            "hidden_nodecay":{"params": [], "lr": hidden_lr, "weight_decay": 0.0},
            "output_decay":  {"params": [], "lr": base_lr,   "weight_decay": weight_decay},
            "output_nodecay":{"params": [], "lr": base_lr,   "weight_decay": 0.0},
        }

        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue
            kind = _classify(name, p)
            has_decay = p.ndim >= 2 and "bias" not in name and "norm" not in name
            key = f"{kind}_{'decay' if has_decay else 'nodecay'}"
            groups[key]["params"].append(p)

        # Drop empty groups (AdamW will warn otherwise)
        return [g for g in groups.values() if len(g["params"]) > 0]

    def forward(self, idx, targets, loss_reduction="mean") -> GPTOutput:
        """
        Training forward pass. Always computes loss.
        targets is required — if you don't have targets, use forward_inference().
        Returns logits over all positions (B, T, vocab_size).
        """
        x = self._encode(idx)
        logits = self._decode(x)

        # Flatten for CrossEntropy: (B*T, Vocab)
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            targets.reshape(-1),
            reduction=loss_reduction
        )
        return GPTOutput(logits=logits, loss=loss)

    def forward_inference(self, idx) -> torch.Tensor:
        """
        Inference forward pass. Returns logits for last token only (B, 1, vocab_size).
        No loss computation. Use this for generation.
        """
        x = self._encode(idx)
        # Only decode the last position — avoids materializing full (B, T, vocab) tensor
        logits = self._decode(x[:, [-1], :])
        return logits