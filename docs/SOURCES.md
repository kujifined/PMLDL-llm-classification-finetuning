# Sources and attribution register

This file is the single register for code, data, models, text, and figures that
were not created by the project team. Add the exact URL, version or revision,
license, affected files, and nature of adaptation before using a source.

## Competition and data

- Kaggle, LLM Classification Finetuning overview:
  https://www.kaggle.com/competitions/llm-classification-finetuning/overview
- Kaggle data page:
  https://www.kaggle.com/competitions/llm-classification-finetuning/data
- Kaggle competition rules:
  https://www.kaggle.com/competitions/llm-classification-finetuning/rules
- Competition dataset license: CC BY-NC 4.0.

## Scientific context

- Chiang et al. Chatbot Arena: An Open Platform for Evaluating LLMs by Human
  Preference. ICML 2024.
  https://proceedings.mlr.press/v235/chiang24b.html
- Zheng et al. Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena. 2023.
  https://arxiv.org/abs/2306.05685

## Candidate baseline sources

- Kaggle pinned LMSYS KerasNLP starter:
  https://www.kaggle.com/code/addisonhoward/lmsys-kerasnlp-starter
- Kaggle community DeBERTa notebook candidate:
  https://www.kaggle.com/code/adelanseur/llm-classification-finetuning-deberta

Status: neither candidate has been copied or adapted. Before reproduction,
record each exact notebook version, verify its displayed license, and map every
local file derived from it. A URL alone is not sufficient attribution.

## Pretrained models

- microsoft/deberta-v3-small (Hugging Face Hub):
  https://huggingface.co/microsoft/deberta-v3-small
  License: MIT. Revision: `main` (no specific commit/revision was pinned for
  the E060 run; pin one on the next re-run for full reproducibility).
  Used as the shared encoder backbone for E060 (exact swap-equivariant pair
  encoder), fine-tuned by the team via `scripts/train_symmetric_encoder.py`;
  no code was copied from a third-party notebook for this experiment.
