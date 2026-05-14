import os
import json
import math
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.amp import GradScaler, autocast
from transformers import AutoTokenizer, AutoModel
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional


# SABİTLER

@dataclass
class SmartChunkerConfig:
    # Encoder
    encoder_name: str    = "intfloat/multilingual-e5-small"
    hidden_size: int     = 384
    max_seq_len: int     = 128
    max_sents: int       = 20
    dropout: float       = 0.1

    # Cross-Sentence Attention
    n_layers: int        = 2
    n_heads: int         = 4

    # Eğitim
    learning_rate: float = 2e-5
    encoder_lr: float    = 1e-5
    weight_decay: float  = 0.01
    warmup_ratio: float  = 0.1
    max_epochs: int      = 10
    batch_size: int      = 16
    grad_clip: float     = 1.0

    # Otomatik hesaplanan
    pos_weight: float    = 10.3
    max_chunk_sents: int = 17
    threshold: float     = 0.5

    checkpoint_dir: str  = "checkpoints/"


def load_pos_weight(stats_path: str, split: str = "train") -> float:
    try:
        with open(stats_path, "r") as f:
            stats = json.load(f)
        ratio      = stats[split]["boundary_ratio"]
        pos_weight = (1.0 - ratio) / ratio
        print(f"[Config] Boundary oranı: {ratio:.2%} → pos_weight: {pos_weight:.2f}")
        return pos_weight
    except Exception as e:
        print(f"[Config] stats.json okunamadı ({e}), varsayılan pos_weight kullanılıyor.")
        return 10.3


def load_max_chunk_sents(
    stats_path: str,
    split: str = "train",
    multiplier: float = 1.5,
) -> int:
    try:
        with open(stats_path, "r") as f:
            stats = json.load(f)
        ratio     = stats[split]["boundary_ratio"]
        avg_chunk = 1.0 / ratio
        max_chunk = int(avg_chunk * multiplier)
        print(f"[Config] Ort. chunk: {avg_chunk:.1f} max_chunk_sents: {max_chunk}")
        return max_chunk
    except Exception as e:
        print(f"[Config] stats.json okunamadı ({e})")
        return 17


# CROSS-SENTENCE ATTENTION

class SentenceTransformerLayer(nn.Module):

    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()

        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
        )
        self.norm1        = nn.LayerNorm(d_model)
        self.norm2        = nn.LayerNorm(d_model)
        self.dropout      = nn.Dropout(dropout)


    def forward(
        self,
        x: torch.Tensor,
        src_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Pre-norm attention
        residual = x
        x = self.norm1(x)
        x, _ = self.self_attn(x, x, x, key_padding_mask=src_key_padding_mask)
        x = self.dropout(x) + residual

        # Pre-norm FFN
        residual = x
        x = self.norm2(x)
        x = self.ffn(x)
        x = self.dropout(x) + residual

        return x


class CrossSentenceAttention(nn.Module):

    def __init__(self, config: SmartChunkerConfig):
        super().__init__()

        self.pos_embedding = nn.Embedding(config.max_sents, config.hidden_size)
        self.layers = nn.ModuleList([
            SentenceTransformerLayer(
                d_model=config.hidden_size,
                n_heads=config.n_heads,
                dropout=config.dropout,
            )
            for _ in range(config.n_layers)
        ])

    def forward(
        self,
        x: torch.Tensor,
        n_sentences: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        B, N, _ = x.shape
        positions = torch.arange(N, device=x.device).unsqueeze(0).expand(B, -1)
        x = x + self.pos_embedding(positions)

        src_key_padding_mask = None
        if n_sentences is not None:
            src_key_padding_mask = torch.zeros(B, N, dtype=torch.bool, device=x.device)
            for i, n in enumerate(n_sentences):
                src_key_padding_mask[i, n:] = True

        for layer in self.layers:
            x = layer(x, src_key_padding_mask)

        return x


# ANA MODEL

class SmartChunker(nn.Module):

    def __init__(self, config: SmartChunkerConfig):
        super().__init__()
        self.config = config

        self.encoder         = AutoModel.from_pretrained(config.encoder_name)
        self.layer_norm      = nn.LayerNorm(config.hidden_size)
        self.encoder_dropout = nn.Dropout(config.dropout)

        self.context_encoder = CrossSentenceAttention(config)

        self.classifier = nn.Sequential(
            nn.Linear(config.hidden_size * 2, config.hidden_size),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_size, 1),
        )

    def mean_pool(
        self,
        token_embeddings: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        mask   = attention_mask.unsqueeze(-1).float()
        summed = (token_embeddings * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1e-9)
        return summed / counts

    def encode_sentences(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, n_sents, seq_len = input_ids.shape

        flat_ids  = input_ids.view(batch_size * n_sents, seq_len)
        flat_mask = attention_mask.view(batch_size * n_sents, seq_len)

        outputs = self.encoder(input_ids=flat_ids, attention_mask=flat_mask)

        pooled = self.mean_pool(outputs.last_hidden_state, flat_mask)
        pooled = self.encoder_dropout(self.layer_norm(pooled))

        return pooled.view(batch_size, n_sents, -1)

    @staticmethod
    def add_prefix(sentences: List[str], prefix: str = "passage: ") -> List[str]:
        """
        mE5-small için gerekli prefix.
        Eğitimde tokenizer'a geçmeden önce, inference'ta predict() içinde kullanılır.
        """
        return [prefix + s for s in sentences]

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        n_sentences: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:

        # Encode 
        sentence_vecs = self.encode_sentences(input_ids, attention_mask)

        # Cross-Sentence Attention 
        context_vecs = self.context_encoder(sentence_vecs, n_sentences)

        # Komşu çift concat
        left  = context_vecs[:, :-1, :]
        right = context_vecs[:, 1:,  :]
        pairs = torch.cat([left, right], dim=-1)

        # Classification
        logits = self.classifier(pairs).squeeze(-1)
        probs  = torch.sigmoid(logits)

        output = {"logits": logits, "probs": probs}

        # maskeleme
        if labels is not None:
            boundary_labels = labels[:, :-1].float()

            if n_sentences is not None:
                batch_size, n_pairs = logits.shape
                pair_mask = torch.zeros(batch_size, n_pairs, device=logits.device)
                for i, n in enumerate(n_sentences):
                    pair_mask[i, :n - 1] = 1.0
            else:
                pair_mask = torch.ones_like(logits)

            pos_weight = torch.tensor(self.config.pos_weight, device=logits.device)
            loss_fn    = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction="none")
            loss       = (loss_fn(logits, boundary_labels) * pair_mask).sum() / pair_mask.sum().clamp(min=1)
            output["loss"] = loss

        return output

    def _predict_long(
        self,
        sentences: List[str],
        tokenizer,
        threshold: float,
        max_chunk_sents: int,
    ) -> Tuple[List[int], List[float]]:
    
        window     = self.config.max_sents
        stride     = window // 2
        n_sent     = len(sentences)
        all_probs  = [0.0] * (n_sent - 1)
        all_counts = [0]   * (n_sent - 1)
        device     = next(self.parameters()).device

        for start in range(0, n_sent - 1, stride):
            end          = min(start + window, n_sent)
            window_sents = sentences[start:end]

            encoded = tokenizer(
                window_sents,
                padding="max_length",
                truncation=True,
                max_length=self.config.max_seq_len,
                return_tensors="pt",
            )
            ids  = encoded["input_ids"].unsqueeze(0).to(device)
            mask = encoded["attention_mask"].unsqueeze(0).to(device)

            with torch.no_grad():
                output = self.forward(ids, mask)

            for local_i, p in enumerate(output["probs"].squeeze(0).cpu().tolist()):
                global_i = start + local_i
                if global_i < len(all_probs):
                    all_probs[global_i]  += p
                    all_counts[global_i] += 1

        final_probs      = [
            all_probs[i] / all_counts[i] if all_counts[i] > 0 else 0.0
            for i in range(len(all_probs))
        ]
        boundary_indices = [i for i, p in enumerate(final_probs) if p >= threshold]
        return boundary_indices, final_probs

    def predict(
        self,
        sentences: List[str],
        tokenizer,
        threshold: Optional[float] = None,
        max_chunk_sents: Optional[int] = None,
    ) -> Tuple[List[int], List[float]]:

        self.eval()
        if threshold is None:
            threshold = self.config.threshold
        if max_chunk_sents is None:
            max_chunk_sents = self.config.max_chunk_sents

        if len(sentences) > self.config.max_sents:
            return self._predict_long(sentences, tokenizer, threshold, max_chunk_sents)

        device    = next(self.parameters()).device
        sentences = self.add_prefix(sentences)
        encoded   = tokenizer(
            sentences,
            padding="max_length",
            truncation=True,
            max_length=self.config.max_seq_len,
            return_tensors="pt",
        )

        input_ids      = encoded["input_ids"].unsqueeze(0).to(device)
        attention_mask = encoded["attention_mask"].unsqueeze(0).to(device)

        with torch.no_grad():
            output = self.forward(input_ids, attention_mask)

        probs            = output["probs"].squeeze(0).cpu().tolist()
        boundary_indices = [i for i, p in enumerate(probs) if p >= threshold]

        # Fallback
        final_boundaries = []
        prev = -1
        for idx in boundary_indices:
            if idx - prev > max_chunk_sents:
                final_boundaries.append((prev + idx) // 2)
            final_boundaries.append(idx)
            prev = idx

        last = boundary_indices[-1] if boundary_indices else -1
        if (len(sentences) - 1) - last > max_chunk_sents:
            final_boundaries.append((last + len(sentences) - 1) // 2)

        return final_boundaries, probs


#  DATASET

class ChunkingDataset(Dataset):

    def __init__(self, jsonl_path: str, tokenizer, config: SmartChunkerConfig):
        self.tokenizer = tokenizer
        self.config    = config
        self.samples   = []

        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                self.samples.append(json.loads(line))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        sample     = self.samples[idx]
        sentences  = sample["sentences"][:self.config.max_sents]
        boundaries = sample["boundaries"][:self.config.max_sents]
        n          = len(sentences)
        pad_count  = self.config.max_sents - n

        # mE5-small için passage prefix
        sentences = SmartChunker.add_prefix(sentences)
        encoded   = self.tokenizer(
            sentences,
            padding="max_length",
            truncation=True,
            max_length=self.config.max_seq_len,
            return_tensors="pt",
        )

        input_ids      = encoded["input_ids"]
        attention_mask = encoded["attention_mask"]

        if pad_count > 0:
            z = torch.zeros(pad_count, self.config.max_seq_len, dtype=torch.long)
            input_ids      = torch.cat([input_ids, z], dim=0)
            attention_mask = torch.cat([attention_mask, z], dim=0)

        labels = torch.tensor(boundaries + [0] * pad_count, dtype=torch.float)

        return {
            "input_ids":      input_ids,
            "attention_mask": attention_mask,
            "labels":         labels,
            "n_sentences":    n,
        }


# TRAIN

class Trainer:

    def __init__(
        self,
        model: SmartChunker,
        config: SmartChunkerConfig,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: str,
    ):
        self.model        = model.to(device)
        self.config       = config
        self.train_loader = train_loader
        self.val_loader   = val_loader
        self.device       = device
        self.best_val_f1  = 0.0

        self.use_amp = device == "cuda"
        self.scaler  = GradScaler(self.device if self.use_amp else "cpu", enabled=self.use_amp)

        encoder_params = list(model.encoder.parameters())
        head_params    = (
            list(model.context_encoder.parameters()) +
            list(model.classifier.parameters())      +
            list(model.layer_norm.parameters())
        )

        self.optimizer = torch.optim.AdamW([
            {"params": encoder_params, "lr": config.encoder_lr},
            {"params": head_params,    "lr": config.learning_rate},
        ], weight_decay=config.weight_decay)

        total_steps  = len(train_loader) * config.max_epochs
        warmup_steps = int(total_steps * config.warmup_ratio)
        self.scheduler = self._cosine_warmup(total_steps, warmup_steps)

        os.makedirs(config.checkpoint_dir, exist_ok=True)

    def _cosine_warmup(self, total_steps, warmup_steps):
        from torch.optim.lr_scheduler import LambdaLR

        def lr_lambda(step):
            if step < warmup_steps:
                return step / max(1, warmup_steps)
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

        return LambdaLR(self.optimizer, lr_lambda)

    def _metrics(self, probs, labels, threshold=0.5):
        preds     = [1 if p >= threshold else 0 for p in probs]
        tp        = sum(p == 1 and l == 1 for p, l in zip(preds, labels))
        fp        = sum(p == 1 and l == 0 for p, l in zip(preds, labels))
        fn        = sum(p == 0 and l == 1 for p, l in zip(preds, labels))
        precision = tp / (tp + fp + 1e-8)
        recall    = tp / (tp + fn + 1e-8)
        f1        = 2 * precision * recall / (precision + recall + 1e-8)
        return {"precision": precision, "recall": recall, "f1": f1}

    def _find_best_threshold(self, probs, labels) -> float:
        best_f1, best_t = 0.0, 0.5
        for t in [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7]:
            f1 = self._metrics(probs, labels, threshold=t)["f1"]
            if f1 > best_f1:
                best_f1, best_t = f1, t
        return best_t

    def train_epoch(self) -> float:
        self.model.train()
        total_loss = 0.0

        for batch in self.train_loader:
            ids     = batch["input_ids"].to(self.device)
            mask    = batch["attention_mask"].to(self.device)
            labels  = batch["labels"].to(self.device)
            n_sents = batch["n_sentences"]

            self.optimizer.zero_grad()

            with autocast(self.device, enabled=self.use_amp):
                output = self.model(ids, mask, labels, n_sents)
                loss   = output["loss"]

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()

            total_loss += loss.item()

        return total_loss / len(self.train_loader)

    @torch.no_grad()

    def evaluate(self) -> Dict:
        self.model.eval()
        total_loss = 0.0
        all_probs  = []
        all_labels = []

        for batch in self.val_loader:
            ids    = batch["input_ids"].to(self.device)
            mask   = batch["attention_mask"].to(self.device)
            labels = batch["labels"].to(self.device)
            n_list = batch["n_sentences"]

            with autocast(self.device, enabled=self.use_amp):
                output = self.model(ids, mask, labels, n_list)

            total_loss += output["loss"].item()

            for i, n in enumerate(n_list):
                all_probs.extend(output["probs"][i, :n - 1].cpu().tolist())
                all_labels.extend(labels[i, :n - 1].long().cpu().tolist())

        best_threshold              = self._find_best_threshold(all_probs, all_labels)
        self.model.config.threshold = best_threshold
        metrics                     = self._metrics(all_probs, all_labels, threshold=best_threshold)
        metrics["loss"]             = total_loss / len(self.val_loader)
        metrics["threshold"]        = best_threshold
        return metrics

    def train(self):
        print(f"Device: {self.device} | AMP: {self.use_amp}")
        print(f"Train: {len(self.train_loader.dataset)} | Val: {len(self.val_loader.dataset)}\n")

        for epoch in range(1, self.config.max_epochs + 1):
            train_loss = self.train_epoch()
            val        = self.evaluate()

            print(
                f"Epoch {epoch:02d}/{self.config.max_epochs} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val['loss']:.4f} | "
                f"F1: {val['f1']:.4f} | "
                f"P: {val['precision']:.4f} | "
                f"R: {val['recall']:.4f} | "
                f"Threshold: {val['threshold']:.2f}"
            )

            if val["f1"] > self.best_val_f1:
                self.best_val_f1 = val["f1"]
                torch.save({
                    "epoch":            epoch,
                    "model_state_dict": self.model.state_dict(),
                    "metrics":          val,
                    "config":           self.config,
                }, os.path.join(self.config.checkpoint_dir, "best_model.pt"))
                print(f"Kaydedildi (F1: {val['f1']:.4f})")

        print(f"\n En iyi Val F1: {self.best_val_f1:.4f}")


# MAIN

if __name__ == "__main__":
    config = SmartChunkerConfig()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    config.pos_weight      = load_pos_weight("data/stats.json")
    config.max_chunk_sents = load_max_chunk_sents("data/stats.json")

    tokenizer = AutoTokenizer.from_pretrained(config.encoder_name)

    train_dataset = ChunkingDataset("data/train.jsonl", tokenizer, config)
    val_dataset   = ChunkingDataset("data/val.jsonl",   tokenizer, config)

    train_loader = DataLoader(
        train_dataset, batch_size=config.batch_size,
        shuffle=True, num_workers=2, pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config.batch_size,
        shuffle=False, num_workers=2,
    )

    model   = SmartChunker(config)
    trainer = Trainer(model, config, train_loader, val_loader, device)
    trainer.train()
