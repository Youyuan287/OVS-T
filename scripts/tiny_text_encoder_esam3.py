import json
from pathlib import Path

import torch
import torch.nn as nn

from sam3.model.tokenizer_ve import SimpleTokenizer


class TinyTextEncoderESAM3(nn.Module):
    """
    Tiny online text encoder for ESAM3.

    Output format is compatible with ESAM3 backbone.forward_text:
      language_features: [77, B, 256]
      language_embeds:   [77, B, 512]
      language_mask:     [B, 77], True means padding / invalid token
    """

    def __init__(
        self,
        bpe_path="sam3/assets/bpe_simple_vocab_16e6.txt.gz",
        context_length=77,
        token_dim=192,
        hidden_dim=256,
        num_layers=3,
        num_heads=6,
        dropout=0.1,
    ):
        super().__init__()

        self.bpe_path = bpe_path
        self.context_length = int(context_length)
        self.tokenizer = SimpleTokenizer(bpe_path=bpe_path)

        self.vocab_size = int(self.tokenizer.vocab_size)
        self.sot_token_id = int(self.tokenizer.sot_token_id)
        self.eot_token_id = int(self.tokenizer.eot_token_id)

        self.token_embedding = nn.Embedding(self.vocab_size, token_dim)
        self.position_embedding = nn.Parameter(torch.zeros(1, self.context_length, token_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=token_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        self.norm = nn.LayerNorm(token_dim)
        self.to_language_features = nn.Linear(token_dim, 256)
        self.to_language_embeds = nn.Linear(token_dim, 512)

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.position_embedding, std=0.01)
        nn.init.xavier_uniform_(self.to_language_features.weight)
        nn.init.zeros_(self.to_language_features.bias)
        nn.init.xavier_uniform_(self.to_language_embeds.weight)
        nn.init.zeros_(self.to_language_embeds.bias)

    def tokenize(self, prompts, device=None):
        if isinstance(prompts, str):
            prompts = [prompts]

        batch_ids = []
        batch_mask = []

        max_content_len = self.context_length - 2

        for text in prompts:
            ids = self.tokenizer.encode(str(text))
            ids = ids[:max_content_len]
            ids = [self.sot_token_id] + ids + [self.eot_token_id]

            valid_len = len(ids)
            pad_len = self.context_length - valid_len

            ids = ids + [0] * pad_len
            mask = [False] * valid_len + [True] * pad_len

            batch_ids.append(ids)
            batch_mask.append(mask)

        token_ids = torch.tensor(batch_ids, dtype=torch.long, device=device)
        language_mask = torch.tensor(batch_mask, dtype=torch.bool, device=device)

        return token_ids, language_mask

    def forward_text(self, prompts, device=None):
        if device is None:
            device = next(self.parameters()).device

        token_ids, language_mask = self.tokenize(prompts, device=device)

        x = self.token_embedding(token_ids)
        x = x + self.position_embedding[:, : x.shape[1], :].to(x.dtype)

        x = self.encoder(x, src_key_padding_mask=language_mask)
        x = self.norm(x)

        language_features = self.to_language_features(x).transpose(0, 1).contiguous()
        language_embeds = self.to_language_embeds(x).transpose(0, 1).contiguous()

        return {
            "language_features": language_features,
            "language_mask": language_mask,
            "language_embeds": language_embeds,
        }

    def forward(self, prompts, device=None):
        return self.forward_text(prompts, device=device)

    def save_config(self, path):
        cfg = {
            "bpe_path": self.bpe_path,
            "context_length": self.context_length,
            "vocab_size": self.vocab_size,
            "sot_token_id": self.sot_token_id,
            "eot_token_id": self.eot_token_id,
            "token_dim": self.token_embedding.embedding_dim,
        }
        Path(path).write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
