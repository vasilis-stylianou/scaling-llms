import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from scaling_llms.models import (
    Block,
    CausalSelfAttention,
    GPTConfig,
    GPTModel,
    MLP
)


# ============================================================
# TESTS FOR CAUSAL SELF-ATTENTION
# ============================================================
@pytest.fixture
def attn_cfg():
    return GPTConfig(n_embd=32, n_head=4, seq_len=20)


@pytest.fixture
def attn(attn_cfg):
    return CausalSelfAttention(attn_cfg)


def test_attn_output_shape(attn_cfg, attn):
    """Test 1: Shape Integrity"""
    B, T, D = 2, 10, attn_cfg.n_embd
    x = torch.randn(B, T, D)
    y = attn(x)
    assert y.shape == (B, T, D), f"Shape mismatch: Expected {(B, T, D)}, got {y.shape}"


def test_attn_causality(attn_cfg, attn):
    """Test 2: The Leakage Test (Past cannot see Future)"""
    B, T, D = 1, 5, attn_cfg.n_embd
    x = torch.randn(B, T, D)

    # Run 1: Normal
    with torch.no_grad():
        y1 = attn(x)

    # Run 2: Change the LAST token (Future)
    x_perturbed = x.clone()
    x_perturbed[:, -1, :] += 99.0

    with torch.no_grad():
        y2 = attn(x_perturbed)

    # Check FIRST token (Past) - Should be identical
    diff = torch.abs(y1[:, 0, :] - y2[:, 0, :]).sum()
    assert diff < 1e-6, f"Causality Failed! Future change affected past. Diff: {diff}"


def test_attn_overfitting(attn_cfg):
    """Test 3: Sanity Check (Can it learn?)"""
    torch.manual_seed(42)
    model = CausalSelfAttention(attn_cfg)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)

    # Learn to output 0 for everything (Dummy task)
    x = torch.randn(4, 6, attn_cfg.n_embd)
    target = torch.zeros_like(x)

    losses = []
    for _ in range(50):
        optimizer.zero_grad()
        y = model(x)
        loss = F.mse_loss(y, target)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    assert losses[-1] < losses[0], "Model is not learning (Loss did not decrease)"


# ============================================================
# TESTS FOR MLP
# ============================================================
@pytest.fixture
def mlp_cfg():
    return GPTConfig(n_embd=32)


@pytest.fixture
def mlp(mlp_cfg):
    return MLP(mlp_cfg)


def test_mlp_shape_integrity(mlp_cfg, mlp):
    """Test 1: Does (B, T, D) input result in (B, T, D) output?"""
    B, T, D = 2, 10, mlp_cfg.n_embd
    x = torch.randn(B, T, D)
    y = mlp(x)
    assert y.shape == (B, T, D)


def test_mlp_position_independence(mlp_cfg, mlp):
    """
    Test 2: The 'Isolation' Test.
    Unlike Attention, MLP should process every token independently.
    Changing Token 1 should NOT affect the output of Token 2.
    """
    B, T, D = 1, 5, mlp_cfg.n_embd
    x = torch.randn(B, T, D)

    # Run 1
    with torch.no_grad():
        y1 = mlp(x)

    # Perturb Token 0
    x_perturbed = x.clone()
    x_perturbed[:, 0, :] += 10.0  # Change first token massively

    # Run 2
    with torch.no_grad():
        y2 = mlp(x_perturbed)

    # Check Token 1 (Should be identical)
    # We check index 1, not index 0.
    diff = torch.abs(y1[:, 1, :] - y2[:, 1, :]).sum()

    assert diff < 1e-6, (
        f"Independence Failed! Changing Token 0 altered Token 1 output. Diff: {diff}"
    )


def test_mlp_learning_capacity():
    """Test 3: Can it overfit a simple target? (Gradients check)"""
    torch.manual_seed(42)
    cfg = GPTConfig(n_embd=16)
    model = MLP(cfg)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)

    x = torch.randn(4, 5, 16)
    target = torch.randn(4, 5, 16)

    losses = []
    for _ in range(50):
        optimizer.zero_grad()
        y = model(x)
        loss = nn.MSELoss()(y, target)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    assert losses[-1] < losses[0], "MLP failed to learn (Loss didn't decrease)"


# ============================================================
# TESTS FOR BLOCK
# ============================================================
@pytest.fixture
def block_cfg():
    return GPTConfig(n_embd=32, n_head=4, seq_len=20)


@pytest.fixture
def block(block_cfg):
    return Block(block_cfg)


def test_block_output_shape(block_cfg, block):
    """Test 1: Shape Consistency"""
    B, T, D = 2, 10, block_cfg.n_embd
    x = torch.randn(B, T, D)
    y = block(x)
    assert y.shape == (B, T, D)


def test_block_residual_property(block_cfg):
    """
    Test 2: The Gradient Superhighway.
    """
    block = Block(block_cfg)

    # 1. Zero out Weights (so Wx = 0)
    nn.init.zeros_(block.attn.c_proj.weight)
    nn.init.zeros_(block.mlp.c_proj.weight)

    # 2. Zero out Biases (Only if they exist)
    if block.attn.c_proj.bias is not None:
        nn.init.zeros_(block.attn.c_proj.bias)

    if block.mlp.c_proj.bias is not None:
        nn.init.zeros_(block.mlp.c_proj.bias)

    x = torch.randn(2, 5, 32)
    y = block(x)

    diff = torch.abs(x - y).sum()
    assert diff < 1e-6, f"Residual connection broken. Diff: {diff}"


def test_block_gradient_flow(block):
    """Test 3: Do gradients reach both Attention and MLP?"""
    x = torch.randn(2, 5, 32, requires_grad=True)
    y = block(x)
    loss = y.sum()
    loss.backward()

    # Check if gradients exist in sub-components
    assert block.attn.c_attn.weight.grad is not None, "No grad in Attention"
    assert block.mlp.c_fc.weight.grad is not None, "No grad in MLP"
    assert block.ln1.weight.grad is not None, "No grad in LayerNorm 1"


# ============================================================
# TESTS FOR GPT MODEL
# ============================================================
@pytest.fixture
def gpt_cfg():
    return GPTConfig(
        vocab_size=100,
        n_layer=2,
        n_head=2,
        n_embd=32,
        seq_len=20,
        embd_pdrop=0.0,
        resid_pdrop=0.0,
        attn_pdrop=0.0,
    )


@pytest.fixture
def gpt_model(gpt_cfg):
    return GPTModel(gpt_cfg)


def test_gpt_forward_pass_inference(gpt_cfg, gpt_model):
    """Test 1: Inference Mode (No Targets)"""
    B, T = 2, 10
    idx = torch.randint(0, gpt_cfg.vocab_size, (B, T))

    # Should return logits for only the LAST token (optimization)
    out = gpt_model(idx)

    assert out.loss is None
    # Expected: (B, 1, Vocab)
    assert out.logits.shape == (B, 1, gpt_cfg.vocab_size)


def test_gpt_forward_pass_training(gpt_cfg, gpt_model):
    """Test 2: Training Mode (With Targets)"""
    B, T = 2, 10
    idx = torch.randint(0, gpt_cfg.vocab_size, (B, T))
    targets = torch.randint(0, gpt_cfg.vocab_size, (B, T))

    out = gpt_model(idx, targets)

    assert out.loss is not None
    # Expected: (B, T, Vocab) - Full sequence for loss calc
    assert out.logits.shape == (B, T, gpt_cfg.vocab_size)


def test_gpt_weight_tying(gpt_model):
    """Test 3: Are embeddings and head weights the same object?"""
    # Pointers should be identical
    assert gpt_model.transformer.wte.weight is gpt_model.lm_head.weight
