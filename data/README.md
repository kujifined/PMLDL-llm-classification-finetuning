# Competition data

The raw archive and extracted CSV files are intentionally ignored by Git.
They are distributed under the competition terms and CC BY-NC 4.0.

Expected local layout:

~~~text
data/
  llm-classification-finetuning.zip
  llm-classification-finetuning/
    train.csv
    test.csv
    sample_submission.csv
~~~

After accepting the competition rules and configuring the Kaggle API, the
snapshot can be obtained with:

~~~bash
kaggle competitions download -c llm-classification-finetuning -p data
unzip data/llm-classification-finetuning.zip -d data/llm-classification-finetuning
~~~

`scripts/audit_data.py` verifies every entry in checksums.sha256 before reading
the data. `data/splits/folds.csv` is the immutable split used by all local
experiments; its metadata records the source-data and generator versions.

Do not publish or redistribute the raw competition data. Keep any repository
containing private competition artifacts accessible only to the Kaggle team.
