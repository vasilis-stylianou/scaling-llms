from dataclasses import dataclass, field as dc_field
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# -------------------------
# MODEL CONFIGS
# -------------------------
@dataclass
class GPTConfig:
    seq_len: int = 512
    vocab_size: int = 256
    n_embd: int = 256
    n_layer: int = 4
    n_head: int = 4
    tied_embeddings: bool = True

    # optional / overrideable
    d_ff: int | None = None

    embd_pdrop: float = 0.0
    resid_pdrop: float = 0.0
    attn_pdrop: float = 0.0
    norm_type: str = "layernorm"

    # derived (not user inputs)
    d_head: int = dc_field(init=False)

    # --- PRIVATE METHODS ---
    def __post_init__(self):
        # head dimension
        assert self.n_embd % self.n_head == 0, "n_embd must be divisible by n_head"
        self.d_head = self.n_embd // self.n_head

        # MLP expansion
        if self.d_ff is None:
            self.d_ff = 4 * self.n_embd
        else:
            assert self.d_ff > self.n_embd, "d_ff should be larger than n_embd"

        # norm choice
        assert self.norm_type in {"layernorm", "rmsnorm"}, \
            f"Invalid norm type: {self.norm_type}"
        

# -------------------------
# MODEL MODULES
# -------------------------
class CausalSelfAttention(nn.Module):
    
    def __init__(self, cfg):
        super().__init__()
        
        # Key variables
        self.n_head = cfg.n_head
        self.n_embd = cfg.n_embd
        self.d_head = cfg.d_head
        self.seq_len = cfg.seq_len

        # 1. Fused QKV Projection
        self.c_attn = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=False)

        # 2. Output Projection
        self.c_proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)

        # 3. Regularization
        self.attn_dropout = nn.Dropout(cfg.attn_pdrop)
        self.resid_dropout = nn.Dropout(cfg.resid_pdrop)

        # 4. Register Mask as a Buffer 
        # This saves you from passing 'mask' in forward() every time
        self.register_buffer(
            "causal_mask", 
            torch.tril(
                torch.ones(self.seq_len, self.seq_len)
            ).view(1, 1, self.seq_len, self.seq_len)
        )

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

        # --- STEP 2: Attention Mechanism ---
        # (B, n_head, T, d_head) @ (B, n_head, d_head, T) -> (B, n_head, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.d_head))
        
        # Apply Causal Mask (using the registered buffer)
        # NOTE: Slice the mask to T because the current batch may be shorter than cfg.seq_len
        att = att.masked_fill(self.causal_mask[:,:,:T,:T] == 0, float('-inf'))
        
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
    

class MLP(nn.Module):

    def __init__(self, cfg):
        super().__init__()

        # 1. Expansion: n_embd -> d_ff > n_embd
        self.c_fc = nn.Linear(cfg.n_embd, cfg.d_ff) 

        # 2. Activation: GPT-2 used the 'tanh' approximation
        self.gelu = nn.GELU(approximate='tanh')

        # 3. Projection: d_ff -> n_embd
        self.c_proj = nn.Linear(cfg.d_ff, cfg.n_embd)

        # 4. Regularization
        self.dropout = nn.Dropout(cfg.embd_pdrop)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)

        return x


class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        
        # 1. LayerNorms
        #    Note: GPT-2 uses simple LayerNorm. 
        #    (Modern Llama uses RMSNorm, but we stick to GPT-2 here)
        self.ln1 = nn.LayerNorm(cfg.n_embd)
        self.ln2 = nn.LayerNorm(cfg.n_embd)
        
        # 2. Sub-layers
        self.attn = CausalSelfAttention(cfg)
        self.mlp = MLP(cfg)

    def forward(self, x):
        # Weighted sum (Communication)
        # Note the Pre-Norm placement: attn(ln1(x))
        x = x + self.attn(self.ln1(x))
        
        # Element-wise processing (Computation)
        x = x + self.mlp(self.ln2(x))
        
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

        self.transformer = nn.ModuleDict(dict(
            # 1. Embeddings
            wte = nn.Embedding(cfg.vocab_size, cfg.n_embd),
            wpe = nn.Embedding(cfg.seq_len, cfg.n_embd),

            # 2. The Stack (Dropout usually applied after embedding sum)
            drop = nn.Dropout(cfg.resid_pdrop),

            # 3. The Layers (ModuleList allows standard iteration)
            h = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)]),

            # 4. Final Layernorm
            ln_f = nn.LayerNorm(cfg.n_embd),
        ))

        # 5. The Language Modeling Head
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)

        # 6. Weight Tying (Optional but recommended)
        self.transformer.wte.weight = self.lm_head.weight

        # Init weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None, loss_reduction="mean") -> GPTOutput:
        B, T = idx.size()

        # 1. Create Position Indices (0, 1, ..., T-1)
        device = idx.device
        pos = torch.arange(0, T, dtype=torch.long, device=device) # Shape (T)

        # 2. Embeddings
        # Token Emb (B, T, D) + Pos Emb (T, D) -> Broadcasting handles the add
        x = self.transformer.wte(idx) + self.transformer.wpe(pos)
        x = self.transformer.drop(x)

        # 3. Blocks
        for block in self.transformer.h:
            x = block(x)

        # 4. Final Norm
        x = self.transformer.ln_f(x)

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