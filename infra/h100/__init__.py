from __future__ import annotations

from pathlib import Path

import vh3

from ml.nirvana.python_deep_learning.operations.macro_operation import (
    python_3_deep_learning,
)


# This resolves on the machine that invokes VH3 and uploads the local files;
# workers receive them through the two `local_file` inputs below.
INPUT_ROOT = Path.home() / "deberta_peft_ablation_inputs"
CODE_BUNDLE = INPUT_ROOT / "code.tar.gz"
DATA_BUNDLE = INPUT_ROOT / "data.tar.gz"

# The PyDL `pip` parameter takes bare package specifiers and installs them one
# by one; its installer rejects `-r`, `-e` and other pip flags outright. The
# pinned locks are therefore installed by runner.py once the repository has
# been cloned inside the job.


def _deberta_peft_ablation(
    mode: str,
    *,
    trial_id: str | None = None,
    hpo_smoke: bool = False,
) -> None:
    code = vh3.local_file(
        CODE_BUNDLE,
        vh3.Binary,
        name=f"deberta-peft-ablation code ({mode})",
        ttl_days=14,
    )
    data = vh3.local_file(
        DATA_BUNDLE,
        vh3.Binary,
        name="deberta-peft-ablation data and pinned backbone",
        ttl_days=14,
    )
    python_3_deep_learning(
        script=(code,),
        data=(data,),
        pulsar_token=vh3.context.yt_token,
        image=("50729373-e8e6-479c-b610-9f222a9b65c3",),
        python_version="3.12",
        pool_tree=("gpu_hainan_80g",),
        nodes_count=1,
        gpu_count=1,
        run_command="python3.12 $SOURCE_CODE_PATH/runner.py",
        environment=tuple(
            value
            for value in (
                f"PMLDL_RUN_MODE={mode}",
                f"PMLDL_HPO_TRIAL_ID={trial_id}" if trial_id else None,
                "PMLDL_HPO_SMOKE=1" if hpo_smoke else None,
            )
            if value is not None
        ),
        job_scheduler_instance="watt",
        job_scheduler_yt_pool="alice-nlp-functions",
        job_scheduler_yt_custom_spec='{"weight": 2}',
        cpu_cores_usage=1600,
        max_ram=96 * 1024,
        max_disk=30 * 1024,
        ttl=180 if mode == "smoke" else 480,
        collect_telemetry=True,
        propagate_run_command_exit_code=True,
        **vh3.block_args(name=f"deberta-peft-ablation {mode}"),
    )


@vh3.decorator.graph()
def deberta_peft_ablation_smoke() -> None:
    _deberta_peft_ablation("smoke")


@vh3.decorator.graph()
def deberta_peft_ablation_full() -> None:
    _deberta_peft_ablation("full")


@vh3.decorator.graph()
def deberta_qlora_hpo_five() -> None:
    """Run the five fixed QLoRA candidates concurrently on independent H100s."""
    for trial_id in (
        "reference_r16",
        "low_lr_r16",
        "high_lr_r16",
        "rank32",
        "rank32_dropout10",
    ):
        _deberta_peft_ablation("hpo", trial_id=trial_id)


@vh3.decorator.graph()
def deberta_qlora_hpo_smoke() -> None:
    """Exercise the QLoRA HPO runner on a balanced tiny subset."""
    _deberta_peft_ablation(
        "hpo",
        trial_id="reference_r16",
        hpo_smoke=True,
    )
