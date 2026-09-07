from __future__ import annotations

import importlib
import json
import os
import platform
import re
import shutil
import site
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path


EXPERIMENT_ID = "E20260905212645934620"
NOTEBOOK_RELATIVE_PATH = Path(
    "output/jupyter-notebook/"
    "E20260905212645934620__e2-deberta-full-vs-lora-vs.ipynb"
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


# Installed from requirements-transformer.lock, minus two groups:
#   - torch/torchvision/torchaudio: the CUDA image ships its own build and the
#     locked wheels would replace it with one not matched to this driver;
#   - clearml: not published on the internal mirror at any version. The tracker
#     imports it lazily and degrades to status "unavailable", which run
#     validation accepts with a warning.
EXCLUDED_FROM_LOCK = frozenset(
    {"torch", "torchvision", "torchaudio", "clearml"}
)

# transformers 4.56 rewrote checkpoint loading. Its _initialize_missing_keys
# walks every key of model.state_dict(), which for a bitsandbytes 4-bit model
# includes quantisation entries such as "...weight.absmax". Resolving those as
# module paths raises AttributeError: `weight` is not an nn.Module, so the
# QLoRA arm cannot load at all on 4.57.6. 4.56.2 predates that walk while
# still accepting the "dtype" keyword the notebook passes, which 4.55 renamed
# back to "torch_dtype". The
# override applies to every arm so the three of them stay comparable, and the
# realised versions are recorded in the run environment.
LOCK_OVERRIDES = {"transformers": "transformers==4.56.2"}


def lock_requirements(path: Path) -> list[str]:
    requirements: list[str] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        name = re.split(r"[=<>!~\[]", line, maxsplit=1)[0].strip().lower()
        if name in EXCLUDED_FROM_LOCK:
            continue
        requirements.append(LOCK_OVERRIDES.get(name, line))
    return requirements

COMMAND_LOG: list[str] = []


def installed_distributions() -> dict[str, str]:
    from importlib import metadata

    versions: dict[str, str] = {}
    for distribution in metadata.distributions():
        name = distribution.metadata["Name"]
        if name:
            versions[name.lower().replace("_", "-")] = distribution.version
    return versions


def record(text: str) -> None:
    COMMAND_LOG.extend(text.splitlines())


def run_command(arguments: list[str], *, cwd: Path | None = None) -> str:
    record(f"$ {' '.join(arguments)}")
    completed = subprocess.run(
        arguments,
        cwd=cwd,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    print(completed.stdout, flush=True)
    record(completed.stdout)
    if completed.returncode != 0:
        record(f"[exit {completed.returncode}] {' '.join(arguments)}")
        raise subprocess.CalledProcessError(
            completed.returncode, arguments, completed.stdout
        )
    return completed.stdout


def pip_install(arguments: list[str]) -> None:
    """Install into the job interpreter.

    The PyDL `pip` operation parameter cannot be used for this: its installer
    takes bare package specifiers as positional arguments and rejects `-r`,
    `-e` and other pip flags, so requirement files must be installed here,
    after the repository has been cloned.
    """

    base = [sys.executable, "-m", "pip", "install"]
    try:
        run_command(base + arguments)
    except subprocess.CalledProcessError:
        print("Retrying pip install with --user", flush=True)
        run_command(base + ["--user"] + arguments)
        user_site = site.getusersitepackages()
        if user_site not in sys.path:
            sys.path.insert(0, user_site)
    importlib.invalidate_caches()


def ensure_safetensors(model_cache: Path, model_name: str, revision: str) -> Path:
    """Re-serialise the pinned checkpoint as safetensors.

    The pinned revision ships only ``pytorch_model.bin``. Transformers refuses
    to ``torch.load`` it unless torch is at least 2.6 final, and this image
    carries the 2.6.0a0 pre-release, which compares as older. Converting keeps
    the pinned revision and its weights untouched and only changes the file
    format, which that restriction does not apply to.
    """

    folder = "models--" + model_name.replace("/", "--")
    snapshot = model_cache / "hub" / folder / "snapshots" / revision
    target = snapshot / "model.safetensors"
    if target.is_file():
        record(f"safetensors already present: {target}")
        return model_cache

    checkpoint = snapshot / "pytorch_model.bin"
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)

    if not os.access(snapshot, os.W_OK):
        writable = Path(os.environ["DATA_PATH"]) / "hf-home"
        record(f"model cache is read-only, copying to {writable}")
        shutil.copytree(model_cache, writable, dirs_exist_ok=True, symlinks=True)
        return ensure_safetensors(writable, model_name, revision)

    import torch
    from safetensors.torch import save_file

    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    tensors = {name: value.clone().contiguous() for name, value in state.items()}
    save_file(tensors, str(target), metadata={"format": "pt"})
    record(f"converted pytorch_model.bin -> model.safetensors, {len(tensors)} tensors")

    marker = model_cache / "hub" / folder / ".no_exist" / revision / "model.safetensors"
    if marker.is_file():
        marker.unlink()
        record("removed stale .no_exist marker for model.safetensors")
    return model_cache


def main() -> None:
    source_root = Path(os.environ["SOURCE_CODE_PATH"])
    input_root = Path(os.environ["INPUT_DATA_PATH"])
    output_root = Path(os.environ["DATA_PATH"])
    logs_root = Path(os.environ["LOGS_PATH"])
    json_output = Path(os.environ["JSON_OUTPUT_FILE"])
    mode = os.environ.get("PMLDL_RUN_MODE", "smoke").strip().lower()
    if mode not in {"smoke", "full"}:
        raise ValueError(f"Unsupported PMLDL_RUN_MODE={mode!r}")

    output_root.mkdir(parents=True, exist_ok=True)
    logs_root.mkdir(parents=True, exist_ok=True)
    workspace = output_root / "workspace"
    repository = workspace / "PMLDL-llm-classification-finetuning"
    if workspace.exists():
        shutil.rmtree(workspace)
    workspace.mkdir(parents=True)

    bundle = source_root / "pmldl-e2.bundle"
    if not bundle.is_file():
        raise FileNotFoundError(bundle)
    run_command(["git", "clone", str(bundle), str(repository)])

    config_path = repository / "configs" / "experiments" / f"{EXPERIMENT_ID}.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if mode == "smoke":
        config["smoke_test"] = True
        config_path.write_text(
            json.dumps(config, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
            encoding="utf-8",
        )
    elif config.get("smoke_test") is not False:
        raise RuntimeError("The bundled full-run config must have smoke_test=false.")

    competition_dir = input_root / "competition"
    model_cache = input_root / "hf-home"
    required_data = ("train.csv", "test.csv", "sample_submission.csv")
    missing = [name for name in required_data if not (competition_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing competition files: {missing}")
    if not model_cache.is_dir():
        raise FileNotFoundError(model_cache)

    clearml_cache = output_root / "clearml-cache"
    os.environ.update(
        {
            "PMLDL_DATA_DIR": str(competition_dir),
            "HF_HOME": str(model_cache),
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "CLEARML_OFFLINE_MODE": "1",
            "CLEARML_CACHE_DIR": str(clearml_cache),
            "TOKENIZERS_PARALLELISM": "false",
        }
    )

    record(f"sys.executable={sys.executable}")
    record(f"sys.version={sys.version}")
    for probe in (
        [sys.executable, "-m", "pip", "--version"],
        ["python3.12", "-m", "pip", "--version"],
    ):
        try:
            run_command(probe)
        except (subprocess.CalledProcessError, FileNotFoundError) as exc:
            record(f"probe failed: {probe}: {exc}")

    # The pinned locks were resolved against public PyPI and cannot be used
    # here: the internal mirror lags behind and lacks several of those exact
    # versions. The CUDA image already provides a working torch stack, so only
    # the genuinely missing packages are installed, with the image's own torch
    # and numpy held fixed by constraints. The realised environment is recorded
    # in the run artifacts instead of being asserted up front.
    present = installed_distributions()
    record("image packages: " + json.dumps(present, sort_keys=True))
    constraints = [
        f"{name}=={present[name]}"
        for name in ("torch", "torchvision", "torchaudio", "numpy")
        if name in present
    ]
    # The mirror's newest transformers is a 5.x release; this experiment is
    # written against the 4.x API declared in pyproject.toml.
    constraints.append("transformers<5")
    constraints_path = output_root / "constraints.txt"
    constraints_path.write_text("\n".join(constraints) + "\n", encoding="utf-8")
    record(f"constraints: {constraints}")

    requirements = lock_requirements(repository / "requirements-transformer.lock")
    record(f"installing from lock: {requirements}")
    pip_install(["-c", str(constraints_path), *requirements])

    # The repository itself is not pip-installed: building it runs setuptools
    # inside the checkout, which leaves a build/ directory behind and makes the
    # tree dirty. A full experiment refuses to start without a clean commit, so
    # expose the package through PYTHONPATH instead, exactly as the Makefile
    # does. The notebook kernel inherits this environment.
    python_path = str(repository / "src")
    existing = os.environ.get("PYTHONPATH")
    os.environ["PYTHONPATH"] = (
        f"{python_path}{os.pathsep}{existing}" if existing else python_path
    )
    record(f"PYTHONPATH={os.environ['PYTHONPATH']}")
    record("environment after install: " + json.dumps(installed_distributions(), sort_keys=True))

    effective_cache = ensure_safetensors(
        model_cache, config["model"]["name"], config["model"]["revision"]
    )
    if effective_cache != model_cache:
        os.environ["HF_HOME"] = str(effective_cache)

    import nbformat
    import torch
    from nbclient import NotebookClient

    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    environment = {
        "mode": mode,
        "started_at": utc_now(),
        "hostname": platform.node(),
        "python": sys.version,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count(),
        "gpu_name": gpu_name,
        "git_commit": run_command(
            ["git", "rev-parse", "HEAD"], cwd=repository
        ).strip(),
        "yt_pool": "alice-nlp-functions",
        "yt_pool_tree": "gpu_hainan_80g",
        "yt_weight": 2,
    }
    write_json(output_root / "environment.preflight.json", environment)
    print(json.dumps(environment, indent=2, ensure_ascii=False), flush=True)
    if not torch.cuda.is_available() or not gpu_name or "H100" not in gpu_name.upper():
        raise RuntimeError(f"Expected one H100 GPU, got {gpu_name!r}")

    status = run_command(
        ["git", "-C", str(repository), "status", "--porcelain"]
    ).strip()
    record(f"git status before the notebook: {status!r}")

    notebook_path = repository / NOTEBOOK_RELATIVE_PATH
    notebook = nbformat.read(notebook_path, as_version=4)
    executed_path = output_root / f"executed-{mode}.ipynb"
    status: dict[str, object] = {
        **environment,
        "status": "running",
        "executed_notebook": executed_path.name,
    }
    try:
        client = NotebookClient(
            notebook,
            timeout=None,
            kernel_name="python3",
            resources={"metadata": {"path": str(repository)}},
        )
        client.execute()
        status["status"] = "completed"
    except BaseException as exc:
        status["status"] = "failed"
        status["error_type"] = type(exc).__name__
        status["error"] = str(exc)
        (logs_root / "runner-traceback.log").write_text(
            traceback.format_exc(), encoding="utf-8"
        )
        raise
    finally:
        nbformat.write(notebook, executed_path)
        for directory_name in ("results", "artifacts"):
            source = repository / directory_name
            if source.exists():
                shutil.copytree(
                    source,
                    output_root / directory_name,
                    dirs_exist_ok=True,
                )
        status["ended_at"] = utc_now()
        write_json(output_root / "launcher-status.json", status)
        write_json(json_output, status)


def report_diagnostic_failure(exc: BaseException) -> None:
    """Record a failure where it can be read.

    Nirvana keeps no readable copy of this job's stdout, so an unhandled
    exception is otherwise invisible. Write the traceback and the tail of the
    command log to the block outputs, then let the failure propagate so the
    job still reports a non-zero exit.
    """

    payload = {
        "status": "diagnostic_failure",
        "error_type": type(exc).__name__,
        "error": str(exc),
        "traceback": traceback.format_exc().splitlines()[-40:],
        "command_log_tail": COMMAND_LOG[-400:],
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False), flush=True)
    json_output = os.environ.get("JSON_OUTPUT_FILE")
    if json_output:
        write_json(Path(json_output), payload)
    data_path = os.environ.get("DATA_PATH")
    if data_path:
        write_json(Path(data_path) / "diagnostic.json", payload)


if __name__ == "__main__":
    try:
        main()
    except BaseException as error:
        report_diagnostic_failure(error)
        raise
