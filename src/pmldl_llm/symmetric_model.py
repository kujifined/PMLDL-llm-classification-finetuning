
from __future__ import annotations

import torch
from torch import nn
from transformers import AutoModel


class SymmetricPreferenceModel(nn.Module):
    def __init__(self, model_name: str, hidden_dropout: float = 0.1) -> None:
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name).float()
        hidden_size = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(hidden_dropout)

        self.preference_head = nn.Linear(hidden_size, 1, bias=False)

        self.tie_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(hidden_dropout),
            nn.Linear(hidden_size // 2, 1),
        )

    def encode(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:

        output = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden = output.last_hidden_state
        mask = attention_mask.unsqueeze(-1).to(last_hidden.dtype)
        summed = (last_hidden * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1e-6)
        pooled = summed / counts
        return self.dropout(pooled)

    def forward(
        self,
        input_ids_a: torch.Tensor,
        attention_mask_a: torch.Tensor,
        input_ids_b: torch.Tensor,
        attention_mask_b: torch.Tensor,
    ) -> torch.Tensor:
        h_a = self.encode(input_ids_a, attention_mask_a)
        h_b = self.encode(input_ids_b, attention_mask_b)

        diff = h_a - h_b
        abs_diff = torch.abs(diff)

        preference_score = self.preference_head(diff).squeeze(-1)
        tie_score = self.tie_head(abs_diff).squeeze(-1)

        logit_a = preference_score
        logit_b = -preference_score
        logit_tie = tie_score
        return torch.stack([logit_a, logit_b, logit_tie], dim=-1)