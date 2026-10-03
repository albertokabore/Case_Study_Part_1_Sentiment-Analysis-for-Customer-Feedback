"""Bidirectional LSTM sentiment classifier (PyTorch).

Architecture
------------
token ids → embedding (100-d, initialised from Word2Vec trained on the training
tweets) → elementwise dropout → 1-layer BiLSTM (2 × 128 units) → concatenated masked
max- and mean-pooling over time → dropout → linear layer → softmax (3 classes).

Unlike the bag-of-words models, the LSTM reads tokens *in order* (Hochreiter &
Schmidhuber, 1997), so it can model negation scope and phrases. Stop words are
therefore **kept** for this model (word order around "not", "but", "never" matters);
lemmatisation and punctuation removal are still applied.

Training: Adam, cross-entropy with inverse-frequency class weights, gradient
clipping, early stopping on validation macro-F1 (best checkpoint restored).
"""

from __future__ import annotations

import copy
import time
from collections import Counter
from dataclasses import asdict, dataclass

import numpy as np
import torch
from gensim.models import Word2Vec
from sklearn.metrics import f1_score
from sklearn.utils.class_weight import compute_class_weight
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data import DataLoader, TensorDataset

from . import config

PAD, UNK = "<pad>", "<unk>"


@dataclass
class RNNConfig:
    embedding_dim: int = 100
    hidden_dim: int = 128
    dropout: float = 0.4
    max_len: int = 40
    min_freq: int = 2
    batch_size: int = 64
    lr: float = 1e-3
    max_epochs: int = 15
    patience: int = 3
    class_weighted: bool = True
    pretrained_word2vec: bool = True
    seed: int = config.SEED


class Vocabulary:
    """Token ↔ index mapping built from the training set only."""

    def __init__(self, token_lists: list[list[str]], min_freq: int = 2) -> None:
        counts = Counter(tok for tokens in token_lists for tok in tokens)
        self.itos = [PAD, UNK] + sorted(t for t, c in counts.items() if c >= min_freq)
        self.stoi = {t: i for i, t in enumerate(self.itos)}

    def __len__(self) -> int:
        return len(self.itos)

    def encode(self, token_lists: list[list[str]], max_len: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Right-pad / truncate to ``max_len``; returns (ids, true lengths)."""
        ids = np.zeros((len(token_lists), max_len), dtype=np.int64)
        lengths = np.ones(len(token_lists), dtype=np.int64)  # ≥1 for packing
        unk = self.stoi[UNK]
        for i, tokens in enumerate(token_lists):
            seq = [self.stoi.get(t, unk) for t in tokens[:max_len]] or [unk]
            ids[i, : len(seq)] = seq
            lengths[i] = len(seq)
        return torch.from_numpy(ids), torch.from_numpy(lengths)


class BiLSTMClassifier(nn.Module):
    def __init__(self, vocab_size: int, cfg: RNNConfig, n_classes: int = 3,
                 embeddings: np.ndarray | None = None) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, cfg.embedding_dim, padding_idx=0)
        if embeddings is not None:
            self.embedding.weight.data.copy_(torch.from_numpy(embeddings))
        self.embed_dropout = nn.Dropout(cfg.dropout / 2)
        self.lstm = nn.LSTM(cfg.embedding_dim, cfg.hidden_dim, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(cfg.dropout)
        self.fc = nn.Linear(4 * cfg.hidden_dim, n_classes)  # [max ; mean] × 2 directions

    def forward(self, ids: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        emb = self.embed_dropout(self.embedding(ids))
        packed = pack_padded_sequence(emb, lengths.cpu(), batch_first=True, enforce_sorted=False)
        out, _ = self.lstm(packed)
        out, _ = pad_packed_sequence(out, batch_first=True, total_length=ids.size(1))
        mask = (torch.arange(ids.size(1)).unsqueeze(0) < lengths.unsqueeze(1)).unsqueeze(-1)
        max_pool = out.masked_fill(~mask, -1e9).max(dim=1).values
        mean_pool = (out * mask).sum(dim=1) / lengths.unsqueeze(1)
        return self.fc(self.dropout(torch.cat([max_pool, mean_pool], dim=1)))


def _word2vec_matrix(token_lists: list[list[str]], vocab: Vocabulary, cfg: RNNConfig) -> np.ndarray:
    """Train skip-gram Word2Vec on training tokens and align it with ``vocab``."""
    w2v = Word2Vec(token_lists, vector_size=cfg.embedding_dim, window=5, min_count=cfg.min_freq,
                   sg=1, epochs=30, seed=cfg.seed, workers=1)
    rng = np.random.default_rng(cfg.seed)
    matrix = rng.normal(0, 0.1, (len(vocab), cfg.embedding_dim)).astype(np.float32)
    matrix[0] = 0.0  # padding vector
    for i, tok in enumerate(vocab.itos):
        if tok in w2v.wv:
            matrix[i] = w2v.wv[tok]
    return matrix


@torch.no_grad()
def _predict_logits(model: nn.Module, ids: torch.Tensor, lengths: torch.Tensor, batch_size: int = 256) -> torch.Tensor:
    model.eval()
    chunks = [model(ids[i:i + batch_size], lengths[i:i + batch_size]) for i in range(0, len(ids), batch_size)]
    return torch.cat(chunks)


def train_bilstm(train_tokens: list[list[str]], y_train: np.ndarray,
                 val_tokens: list[list[str]], y_val: np.ndarray,
                 cfg: RNNConfig | None = None, verbose: bool = True) -> tuple[nn.Module, Vocabulary, dict]:
    """Train with early stopping; returns (best model, vocabulary, training history)."""
    cfg = cfg or RNNConfig()
    config.set_seed(cfg.seed)
    vocab = Vocabulary(train_tokens, cfg.min_freq)
    embeddings = _word2vec_matrix(train_tokens, vocab, cfg) if cfg.pretrained_word2vec else None
    model = BiLSTMClassifier(len(vocab), cfg, embeddings=embeddings)

    X_tr, L_tr = vocab.encode(train_tokens, cfg.max_len)
    X_va, L_va = vocab.encode(val_tokens, cfg.max_len)
    loader = DataLoader(TensorDataset(X_tr, L_tr, torch.as_tensor(y_train)), batch_size=cfg.batch_size,
                        shuffle=True, generator=torch.Generator().manual_seed(cfg.seed))

    weights = None
    if cfg.class_weighted:
        w = compute_class_weight("balanced", classes=np.arange(len(config.LABELS)), y=y_train)
        weights = torch.tensor(w, dtype=torch.float32)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    history, best_f1, best_state, bad_epochs = [], -1.0, None, 0
    start = time.perf_counter()
    for epoch in range(1, cfg.max_epochs + 1):
        model.train()
        total = 0.0
        for ids, lengths, labels in loader:
            optimizer.zero_grad()
            loss = criterion(model(ids, lengths), labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += loss.item() * len(labels)
        val_logits = _predict_logits(model, X_va, L_va)
        val_loss = criterion(val_logits, torch.as_tensor(y_val)).item()
        val_f1 = f1_score(y_val, val_logits.argmax(1).numpy(), average="macro")
        history.append({"epoch": epoch, "train_loss": total / len(y_train), "val_loss": val_loss, "val_macro_f1": val_f1})
        if verbose:
            print(f"  epoch {epoch:2d}  train_loss={total / len(y_train):.4f}  val_loss={val_loss:.4f}  val_macro_f1={val_f1:.4f}")
        if val_f1 > best_f1:
            best_f1, best_state, bad_epochs = val_f1, copy.deepcopy(model.state_dict()), 0
        else:
            bad_epochs += 1
            if bad_epochs >= cfg.patience:
                break

    model.load_state_dict(best_state)
    info = {
        "config": asdict(cfg),
        "vocab_size": len(vocab),
        "best_val_macro_f1": best_f1,
        "epochs_trained": len(history),
        "best_epoch": int(np.argmax([h["val_macro_f1"] for h in history]) + 1),
        "n_parameters": sum(p.numel() for p in model.parameters()),
        "training_seconds": round(time.perf_counter() - start, 1),
        "history": history,
    }
    return model, vocab, info


def predict_proba(model: nn.Module, vocab: Vocabulary, token_lists: list[list[str]], max_len: int = 40) -> np.ndarray:
    ids, lengths = vocab.encode(token_lists, max_len)
    return torch.softmax(_predict_logits(model, ids, lengths), dim=1).numpy()
