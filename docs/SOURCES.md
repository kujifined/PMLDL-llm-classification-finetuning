# External sources and adaptation record

This file records external material consulted for the E2 and competitive tracks.
No external competition data, pseudo-labels, distilled labels, or notebook source
code is included in this repository.

## Competition and model documentation

- Kaggle, **LLM Classification Finetuning**, ongoing Getting Started code
  competition: <https://www.kaggle.com/competitions/llm-classification-finetuning>.
  The official competition CSV files are used under CC BY-NC 4.0 and remain
  untracked. The rules were accepted by Karim's Kaggle account on 2026-09-06.
- Microsoft, **DeBERTa-v3-base**:
  <https://huggingface.co/microsoft/deberta-v3-base>. E2 pins model commit
  `8ccc9b6f36199bec6961081d44eb72fb3f7353f3`.
- Google, **Gemma-2-9B-IT**:
  <https://huggingface.co/google/gemma-2-9b-it>. The competitive experiment must
  record an immutable model commit before its managed notebook is generated.
- Hugging Face, **Transformers on Apple Silicon**:
  <https://huggingface.co/docs/transformers/perf_train_special>.
- Hugging Face, **bitsandbytes hardware compatibility**:
  <https://huggingface.co/docs/bitsandbytes/en/installation>.

## Scientific context

- Chiang et al., **Chatbot Arena: An Open Platform for Evaluating LLMs by
  Human Preference**, ICML 2024:
  <https://proceedings.mlr.press/v235/chiang24b.html>.
- Zheng et al., **Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena**,
  2023: <https://arxiv.org/abs/2306.05685>.

## Historical solution research

- Chris Deotte, **16th Place - So Close to Gold Medal**, 2024, Kaggle write-up:
  <https://www.kaggle.com/competitions/lmsys-chatbot-arena/writeups/chris-deotte-no-leak-16th-place-so-close-to-gold-m>.
  Adapted ideas: `gemma-2-9b-it`, QLoRA for 2xT4, rank 64, all projection/MLP
  linear modules, one-epoch training, head-tail truncation and swap-time
  averaging. No notebook code was copied.
- Chris Deotte, **16th Place - Infer (3 of 3)**, Kaggle notebook version 2 of 2,
  Apache 2.0:
  <https://www.kaggle.com/code/cdeotte/16th-place-infer-3-of-3>.
  Consulted for the feasibility of offline 2xT4 inference. No code was copied.
- Eisuke Mizutani, **[Training] Gemma-2 9b 4-bit QLoRA fine-tuning**, Kaggle
  notebook version 1 of 1:
  <https://www.kaggle.com/code/emiz6413/training-gemma-2-9b-4-bit-qlora-fine-tuning>.
  Consulted for public 4-bit sequence-classification practice and package-version
  failure reports. No code was copied.
- BlackPearl team, **1st Place Solution - Distill is all you need**, 2024:
  <https://www.kaggle.com/competitions/lmsys-chatbot-arena/writeups/blackpearl-no-leak-1st-place-solution-distill-is-a>.
  The teacher-distillation approach was deliberately not adopted because it is
  outside the agreed data and compute budget.
- Rist team, **2nd place solution**, 2024:
  <https://www.kaggle.com/competitions/lmsys-chatbot-arena/writeups/rist-pref-no-leak-2nd-place-solution>.
  Full-parameter 9B training was deliberately not adopted because it required
  at least two A100-class GPUs.

## Candidate baseline sources retained from the project contract

- Kaggle pinned LMSYS KerasNLP starter:
  <https://www.kaggle.com/code/addisonhoward/lmsys-kerasnlp-starter>.
- Kaggle community DeBERTa notebook candidate:
  <https://www.kaggle.com/code/adelanseur/llm-classification-finetuning-deberta>.

Neither candidate was copied or adapted. A future reproduction must first
record its exact notebook version, displayed license, and every affected local
file; the URL alone is not sufficient attribution.

## Deliberate deviations

- NVIDIA T4 does not provide native BF16 tensor-core training. The notebooks
  request BF16 when supported and fall back to FP16 on T4, recording both the
  requested and actual dtype in `environment.json`. This avoids claiming a BF16
  comparison that the selected hardware cannot execute faithfully.
- Folds 8 and 9 remain unopened. Competitive refit may use folds 0-7 only until
  the Team Lead explicitly unlocks calibration and final-holdout evaluation.
