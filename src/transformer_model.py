"""Fine-tuned Transformer sentiment classifier (DistilRoBERTa).

``distilroberta-base`` (82 M parameters) is a 6-layer distilled version of RoBERTa
(Liu et al., 2019; Sanh et al., 2019). It was chosen over full-size BERT/RoBERTa
because it keeps most of their accuracy at about half the inference cost, which
matters for a model meant to score live customer feedback, and because it can be
fine-tuned on a CPU in under an hour.

Input is the *lightly* cleaned tweet (``preprocessing.light_clean``): the model's
byte-level BPE tokenizer and self-attention exploit casing, punctuation, emoji and
function words, so the aggressive classical preprocessing is deliberately skipped.

Training strategy (standard fine-tuning recipe, Devlin et al., 2019):
AdamW (lr 2e-5, weight decay 0.01), linear warm-up over 10 % of steps then linear
decay, batch size 32, max 64 sub-word tokens, 3 epochs, dynamic padding, best
epoch selected on validation macro-F1.
"""

from __future__ import annotations

import copy
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    get_linear_schedule_with_warmup,
)

from . import config


@dataclass
class TransformerConfig:
    model_name: str = "distilroberta-base"
    max_length: int = 64
    batch_size: int = 32
    lr: float = 2e-5
    weight_decay: float = 0.01
    epochs: int = 3
    warmup_ratio: float = 0.1
    seed: int = config.SEED


def _encode(tokenizer, texts: list[str], labels: np.ndarray | None, max_length: int) -> list[dict]:
    enc = tokenizer(list(texts), truncation=True, max_length=max_length)
    rows = [{k: enc[k][i] for k in enc.keys()} for i in range(len(texts))]
    if labels is not None:
        for row, label in zip(rows, labels):
            row["labels"] = int(label)
    return rows


@torch.no_grad()
def predict_proba(model, tokenizer, texts: list[str], max_length: int = 64, batch_size: int = 64) -> np.ndarray:
    """Class probabilities; batches are length-sorted to minimise padding."""
    model.eval()
    rows = _encode(tokenizer, texts, None, max_length)
    order = np.argsort([len(r["input_ids"]) for r in rows])
    collator = DataCollatorWithPadding(tokenizer)
    probs = np.zeros((len(rows), model.config.num_labels), dtype=np.float32)
    for start in range(0, len(rows), batch_size):
        idx = order[start:start + batch_size]
        batch = collator([rows[i] for i in idx])
        probs[idx] = torch.softmax(model(**batch).logits, dim=-1).numpy()
    return probs


def train_transformer(train_texts: list[str], y_train: np.ndarray,
                      val_texts: list[str], y_val: np.ndarray,
                      cfg: TransformerConfig | None = None,
                      save_dir: Path | None = None, verbose: bool = True):
    """Fine-tune and return (best model, tokenizer, info dict)."""
    cfg = cfg or TransformerConfig()
    config.set_seed(cfg.seed)
    torch.set_num_threads(max(1, torch.get_num_threads()))

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg.model_name, num_labels=len(config.LABELS),
        id2label=config.ID2LABEL, label2id=config.LABEL2ID,
    )
    train_rows = _encode(tokenizer, train_texts, y_train, cfg.max_length)
    loader = DataLoader(train_rows, batch_size=cfg.batch_size, shuffle=True,
                        collate_fn=DataCollatorWithPadding(tokenizer),
                        generator=torch.Generator().manual_seed(cfg.seed))

    no_decay = ("bias", "LayerNorm.weight")
    grouped = [
        {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         "weight_decay": cfg.weight_decay},
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
         "weight_decay": 0.0},
    ]
    optimizer = torch.optim.AdamW(grouped, lr=cfg.lr)
    total_steps = len(loader) * cfg.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, int(cfg.warmup_ratio * total_steps), total_steps)

    history, best_f1, best_state = [], -1.0, None
    start = time.perf_counter()
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        running = 0.0
        for step, batch in enumerate(loader, 1):
            loss = model(**batch).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            running += loss.item()
            if verbose and step % 50 == 0:
                print(f"  epoch {epoch} step {step}/{len(loader)} loss={running / step:.4f} "
                      f"({time.perf_counter() - start:.0f}s)", flush=True)
        val_probs = predict_proba(model, tokenizer, val_texts, cfg.max_length)
        val_f1 = f1_score(y_val, val_probs.argmax(1), average="macro")
        val_loss = float(-np.log(val_probs[np.arange(len(y_val)), y_val] + 1e-12).mean())
        history.append({"epoch": epoch, "train_loss": running / len(loader), "val_loss": val_loss, "val_macro_f1": val_f1})
        if verbose:
            print(f"  epoch {epoch} done: train_loss={running / len(loader):.4f} val_loss={val_loss:.4f} "
                  f"val_macro_f1={val_f1:.4f}", flush=True)
        if val_f1 > best_f1:
            best_f1, best_state = val_f1, copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(save_dir)
        tokenizer.save_pretrained(save_dir)

    info = {
        "config": asdict(cfg),
        "best_val_macro_f1": best_f1,
        "best_epoch": int(np.argmax([h["val_macro_f1"] for h in history]) + 1),
        "n_parameters": sum(p.numel() for p in model.parameters()),
        "training_seconds": round(time.perf_counter() - start, 1),
        "history": history,
    }
    return model, tokenizer, info
