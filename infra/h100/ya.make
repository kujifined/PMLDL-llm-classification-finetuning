PY3_PROGRAM(deberta_peft_ablation)

PY_MAIN(vh3.cli:main)

PEERDIR(
    nirvana/vh3/src
    ml/nirvana/python_deep_learning/operations
)

PY_SRCS(
    __init__.py
)

INCLUDE(${ARCADIA_ROOT}/nirvana/vh3/add_conf.inc)

END()
