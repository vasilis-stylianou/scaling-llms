import pytest
import torch
from dataclasses import asdict
from torch.utils.data import DataLoader, TensorDataset

from scaling_llms.trainer import Trainer, TrainerConfig
from scaling_llms.models import GPTModel, GPTConfig
from scaling_llms.tracking.registries import RunManager, log_as_json
from scaling_llms.constants import RUN_DIRS, RUN_FILES, METRIC_CATS


VOCAB_SIZE = 100
SEQ_LEN = 16
N_EMBD = 32
N_LAYER = 2
N_HEAD = 2
NUM_STEPS = 5
gpt_cfg = GPTConfig(
    vocab_size=VOCAB_SIZE,
    seq_len=SEQ_LEN,
    n_embd=N_EMBD,
    n_layer=N_LAYER,
    n_head=N_HEAD,
)


# ============================================================
# FIXTURES
# ============================================================
@pytest.fixture
def dummy_model():
    """Small model for fast testing."""
    torch.manual_seed(1234)
    return GPTModel(gpt_cfg)


@pytest.fixture
def dummy_dataloader():
    """Minimal dataloader for testing."""
    # Create dummy data: (idx, targets)
    torch.manual_seed(1234)
    batch_size = 4
    num_samples = 20
    
    idx = torch.randint(0, VOCAB_SIZE, (num_samples, SEQ_LEN))
    targets = torch.randint(0, VOCAB_SIZE, (num_samples, SEQ_LEN))
    
    dataset = TensorDataset(idx, targets)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)


@pytest.fixture
def minimal_trainer_config():
    """Minimal valid TrainerConfig for CPU training."""
    return TrainerConfig(
        num_steps=NUM_STEPS,
        lr=1e-3,
        device="cpu",
    )


@pytest.fixture
def extended_trainer_config():
    """TrainerConfig with logging/scheduler features enabled."""
    return TrainerConfig(
        num_steps=NUM_STEPS,
        lr=1e-3,
        device="cpu",
        eval_log_freq=1,
        ckpt_log_freq=1,
        lr_schedule="linear",
        warmup_steps=1,
        min_lr_ratio=0.1,
    )


@pytest.fixture
def tmp_run(tmp_path):
    """Create a temporary run directory for Trainer tests."""
    return RunManager.create_new_run_dir(tmp_path / "runs")


@pytest.fixture
def trainer(minimal_trainer_config, dummy_model, dummy_dataloader, tmp_run):
    """Create a default Trainer instance for tests."""
    return Trainer(
        cfg=minimal_trainer_config,
        model=dummy_model,
        train_dl=dummy_dataloader,
        eval_dl=None,
        run=tmp_run,
    )


@pytest.fixture
def extended_trainer(extended_trainer_config, dummy_model, dummy_dataloader, tmp_path):
    """Trainer with eval, checkpointing, and LR scheduling enabled."""
    run = RunManager.create_new_run_dir(tmp_path / "runs")
    return Trainer(
        cfg=extended_trainer_config,
        model=dummy_model,
        train_dl=dummy_dataloader,
        eval_dl=dummy_dataloader,
        run=run,
    )


# ============================================================
# TESTS FOR TRAINERCONFIG
# ============================================================
def test_trainer_config_init():
    """Test TrainerConfig initializes with required parameters."""
    cfg = TrainerConfig(num_steps=100, lr=3e-4)
    
    # Check required params
    assert cfg.num_steps == 100
    assert cfg.lr == 3e-4
    
    # Check defaults are set
    assert cfg.beta1 == 0.9
    assert cfg.beta2 == 0.95
    assert cfg.weight_decay == 0.1
    assert cfg.accum_steps == 1
    assert cfg.device in ("cpu", "cuda")
    
    # Check post_init configuration ran
    assert cfg.device_name is not None
    assert cfg.precision in ("fp32", "fp16", "bf16")


def test_trainer_config_device_cpu():
    """Test CPU device configuration."""
    cfg = TrainerConfig(num_steps=10, lr=1e-3, device="cpu")
    
    assert cfg.device == "cpu"
    assert cfg.precision == "fp32", "CPU should force FP32"
    assert cfg.device_name == "cpu"


def test_trainer_config_lr_scheduler_validation():
    """Test learning rate scheduler validation."""
    # Valid schedules
    for schedule in ["none", "cosine", "linear"]:
        cfg = TrainerConfig(num_steps=100, lr=1e-3, lr_schedule=schedule)
        assert cfg.lr_schedule == schedule
    
    # Invalid schedule
    with pytest.raises(ValueError, match="lr_schedule must be one of"):
        TrainerConfig(num_steps=100, lr=1e-3, lr_schedule="invalid")


def test_trainer_config_validation_errors():
    """Test TrainerConfig raises errors for invalid parameters."""
    # num_steps must be > 0
    with pytest.raises(ValueError, match="num_steps must be > 0"):
        TrainerConfig(num_steps=0, lr=1e-3)
    
    # lr must be > 0
    with pytest.raises(ValueError, match="lr must be > 0"):
        TrainerConfig(num_steps=10, lr=0)
    
    # warmup_steps must be >= 0
    with pytest.raises(ValueError, match="warmup_steps must be >= 0"):
        TrainerConfig(num_steps=100, lr=1e-3, warmup_steps=-1)
    
    # warmup_steps must be <= num_steps
    with pytest.raises(ValueError, match="warmup_steps must be <= num_steps"):
        TrainerConfig(num_steps=100, lr=1e-3, warmup_steps=200)
    
    # min_lr_ratio must be in [0,1]
    with pytest.raises(ValueError, match="min_lr_ratio must be in"):
        TrainerConfig(num_steps=100, lr=1e-3, min_lr_ratio=1.5)


def test_trainer_config_json_roundtrip(tmp_path):
    """Test TrainerConfig can be saved and loaded from JSON."""
    # Create a config with non-default values
    original_cfg = TrainerConfig(
        num_steps=500,
        lr=5e-4,
        beta1=0.85,
        beta2=0.99,
        weight_decay=0.05,
        precision="bf16",
        accum_steps=4,
        grad_clip_norm=0.5,
        device="cpu",
        lr_schedule="cosine",
        warmup_steps=50,
        min_lr_ratio=0.1,
        enable_tb=True,
        net_log_freq=25,
        sys_log_freq=50,
        eval_log_freq=100,
        ckpt_log_freq=200,
        enable_cuda_timer=True,
        seed=42,
    )
    
    # Save to JSON
    json_path = tmp_path / "trainer_config.json"
    log_as_json(original_cfg, json_path)
    
    assert json_path.exists(), "JSON file should be created"
    
    # Load from JSON
    loaded_cfg = TrainerConfig.from_json(json_path)
    
    # Check all fields match (excluding device_name which is set in __post_init__)
    original_dict = asdict(original_cfg)
    loaded_dict = asdict(loaded_cfg)
    
    for key in original_dict:
        if key != "device_name":  # device_name is set in __post_init__
            assert loaded_dict[key] == original_dict[key], (
                f"Field '{key}' mismatch: {loaded_dict[key]} != {original_dict[key]}"
            )
    
    # device_name should be set by __post_init__
    assert loaded_cfg.device_name is not None


def test_trainer_config_token_budget_derivation_and_json_roundtrip(tmp_path):
    """When num_steps is None, derive it from a token budget and roundtrip via JSON."""
    # Choose values where tokens_per_step = micro_batch_size * seq_len * accum_steps
    train_tokens_budget = 10000
    micro_batch_size = 4
    seq_len = 16
    accum_steps = 2

    original_cfg = TrainerConfig(
        num_steps=None,
        lr=1e-3,
        device="cpu",
        train_tokens_budget=train_tokens_budget,
        micro_batch_size=micro_batch_size,
        seq_len=seq_len,
        accum_steps=accum_steps,
    )

    expected_tokens_per_step = micro_batch_size * seq_len * accum_steps
    expected_num_steps = (train_tokens_budget + expected_tokens_per_step - 1) // expected_tokens_per_step

    assert original_cfg.num_steps == expected_num_steps

    # JSON roundtrip should preserve the derived fields
    json_path = tmp_path / "trainer_config_budget.json"
    log_as_json(original_cfg, json_path)
    loaded_cfg = TrainerConfig.from_json(json_path)

    assert loaded_cfg.num_steps == original_cfg.num_steps


def test_iter_mode_single_batch(dummy_model, dummy_dataloader):
    """`single-batch` mode should always return the same batch from the iterator."""
    cfg = TrainerConfig(num_steps=1, lr=1e-3, device="cpu", iter_mode="single-batch")
    trainer = Trainer(cfg=cfg, model=dummy_model, train_dl=dummy_dataloader, eval_dl=None, run=None)

    b1 = next(trainer.train_iter)
    b2 = next(trainer.train_iter)

    assert torch.equal(b1[0], b2[0])
    assert torch.equal(b1[1], b2[1])


def test_iter_mode_infinite(dummy_model, dummy_dataloader):
    """`infinite` mode should cycle through batches (consecutive yields differ)."""
    cfg = TrainerConfig(num_steps=2, lr=1e-3, device="cpu", iter_mode="infinite")
    trainer = Trainer(cfg=cfg, model=dummy_model, train_dl=dummy_dataloader, eval_dl=None, run=None)

    b1 = next(trainer.train_iter)
    b2 = next(trainer.train_iter)

    # With the dummy_dataloader (batch_size=4, num_samples=20) consecutive
    # batches should typically differ when iter_mode="infinite".
    assert not (torch.equal(b1[0], b2[0]) and torch.equal(b1[1], b2[1]))


# ============================================================
# TESTS FOR TRAINER INITIALIZATION
# ============================================================
def test_trainer_init_state(minimal_trainer_config, dummy_model, dummy_dataloader, tmp_run, trainer):

    assert trainer.cfg is minimal_trainer_config
    assert trainer.model is dummy_model
    assert trainer.train_dl is dummy_dataloader
    assert trainer.eval_dl is None
    assert trainer.run is tmp_run
    assert trainer.device == minimal_trainer_config.device

    assert trainer.train_iter is not None
    assert hasattr(trainer.train_iter, "__next__")

    assert trainer.scaler is not None
    assert isinstance(trainer.optimizer, torch.optim.AdamW)
    assert trainer.lr_scheduler is None
    assert trainer.ckpt_manager is not None
    assert trainer.logger is not None
    assert trainer.logger.name == "Trainer"

    assert trainer.step_idx == 0
    assert trainer.tokens_seen_total == 0
    assert trainer.wall_timer is not None
    assert trainer.cuda_timer is None


def test_trainer_train_resume_logic(trainer):

    # Train for NUM_STEPS
    trainer.train() 
    curr_step_idx = trainer.step_idx
    assert curr_step_idx == NUM_STEPS
    assert trainer.tokens_seen_total > 0

    # Resume training for another (max_steps - NUM_STEPS)
    # CASE 1: max_steps <= NUM_STEPS
    with pytest.raises(ValueError, match="Training already complete"):
        trainer.train(max_steps=NUM_STEPS)

    # CASE 2: max_steps > NUM_STEPS
    max_steps = 2 * NUM_STEPS - 1 
    trainer.train(max_steps) 
    assert (trainer.step_idx - curr_step_idx) == (max_steps - NUM_STEPS)


def test_trainer_checkpoint_roundtrip(minimal_trainer_config, dummy_model, dummy_dataloader, tmp_run, trainer):

    trainer.run.log_metadata(minimal_trainer_config, RUN_FILES.trainer_config, format="json")
    trainer.train()

    ckpt_path = trainer.save_checkpoint("roundtrip.pt")
    assert ckpt_path.exists()

    new_model = GPTModel(gpt_cfg)
    resumed = Trainer.from_checkpoint(
        tmp_run,
        "roundtrip.pt",
        new_model,
        train_dl=dummy_dataloader,
        eval_dl=None,
        strict=True,
    )

    assert resumed.step_idx == trainer.step_idx
    assert resumed.tokens_seen_total == trainer.tokens_seen_total

    for p1, p2 in zip(trainer.model.parameters(), resumed.model.parameters()):
        assert torch.allclose(p1, p2)


def test_trainer_state_dict_roundtrip(minimal_trainer_config, dummy_model, dummy_dataloader, tmp_run):
    trainer = Trainer(
        cfg=minimal_trainer_config,
        model=dummy_model,
        train_dl=dummy_dataloader,
        eval_dl=None,
        run=tmp_run,
    )

    trainer.step_idx = 7
    trainer.tokens_seen_total = 1234

    state = trainer.state_dict()
    new_trainer = Trainer(
        cfg=minimal_trainer_config,
        model=dummy_model,
        train_dl=dummy_dataloader,
        eval_dl=None,
        run=tmp_run,
    )
    new_trainer.load_state_dict(state)

    assert new_trainer.step_idx == 7
    assert new_trainer.tokens_seen_total == 1234


def test_trainer_eval_logging_creates_metrics(extended_trainer):
    trainer = extended_trainer

    trainer.train(max_steps=2)

    eval_metrics_path = trainer.run[RUN_DIRS.metrics] / f"{METRIC_CATS.eval}.jsonl"
    assert eval_metrics_path.exists()
    assert eval_metrics_path.read_text().strip() != ""


def test_trainer_checkpointing_writes_files(extended_trainer):
    trainer = extended_trainer

    trainer.train(max_steps=2) # ckpt_log_freq = 1

    last_ckpt = trainer.run[RUN_DIRS.checkpoints] / RUN_FILES.last_ckpt
    best_ckpt = trainer.run[RUN_DIRS.checkpoints] / RUN_FILES.best_ckpt
    assert last_ckpt.exists()
    assert best_ckpt.exists()


def test_trainer_lr_scheduler_updates_lr(extended_trainer):
    trainer = extended_trainer

    start_lr = trainer.optimizer.param_groups[0]["lr"]
    trainer.train(max_steps=3)
    end_lr = trainer.optimizer.param_groups[0]["lr"]

    assert end_lr != start_lr