"""Paste this single cell into the generated experiment notebook."""

from pmldl_llm.frozen_embeddings import run_frozen_embeddings_experiment


def train_and_evaluate(run):
    return run_frozen_embeddings_experiment(run, SETUP, PROJECT_ROOT)
