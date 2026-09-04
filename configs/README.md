# Experiment configuration

`project.json` is the team-wide contract. Only the Team Lead changes the active
evaluation stage, required metrics, frozen validation paths, or ClearML project.

`experiment.schema.json` defines the fields that every experiment configuration
must contain. Copy `experiments/template.json` for a new hypothesis; never edit
the template in place.

An `experiment_id` identifies a hypothesis. A later experiment runner will add
a unique `run_id` for every concrete execution, so rerunning a seed never
overwrites earlier evidence.

During model selection, every team member must keep:

- `evaluation_role` set to `selection`;
- `smoke_test` set to `true` until the short run succeeds;
- one principal change in `changed_factor`;
- a `parent_experiment_id` for controlled comparisons;
- ClearML metric names and tags from `project.json`.

The Team Lead assigns experiment identifiers to avoid collisions.

The shared ClearML project is `PMLDL LLM Classification Finetuning`. Each team
member creates personal API credentials in the ClearML workspace and keeps them
outside Git. `.env.example` lists the supported variables without secrets.
