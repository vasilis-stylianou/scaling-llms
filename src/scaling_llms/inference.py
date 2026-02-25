import torch
import torch.nn.functional as F

@torch.inference_mode()
def generate(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 200,
    temperature: float = 1.0,
    top_k: int | None = None,
    greedy: bool = False,
    device: torch.device | None = None,
):
    model.eval()
    device = device or next(model.parameters()).device
    seq_len = model.cfg.seq_len
    encode_fn = tokenizer["encode"]
    decode_fn = tokenizer["decode"]

    # Encode prompt and move to device
    input_ids = torch.tensor(encode_fn(prompt), dtype=torch.long, device=device).unsqueeze(0)

    # Iteratively generate tokens auto-regressively
    for _ in range(max_new_tokens):
        # Condition on the most recent seq_len tokens, as context for the next token
        input_cond = input_ids[:, -seq_len:] # shape: (1, seq_len) or (1, <seq_len if prompt shorter)

        # Get logits for the next token
        logits = model(input_cond).logits[:, -1, :] # shape: (1, vocab_size)

        if greedy:
            next_token = torch.argmax(logits, dim=-1, keepdim=True) # shape: (1, 1)
        else:
            logits = logits / max(temperature, 1e-8) 

            if top_k is not None:
                v, _ = torch.topk(logits, top_k) # shape: (1, top_k)
                logits = logits.masked_fill(logits < v[:, [-1]], -float("inf")) # shape: (1, vocab_size)

            probs = F.softmax(logits, dim=-1) # shape: (1, vocab_size)
            next_token = torch.multinomial(probs, num_samples=1) # shape: (1, 1)

        input_ids = torch.cat([input_ids, next_token], dim=1) # shape: (1, seq_len + 1)

    return decode_fn(input_ids[0].tolist()) # decode to string and return