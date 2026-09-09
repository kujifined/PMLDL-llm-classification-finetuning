## Что изменено

Кратко опишите изменение и зачем оно нужно.

## Контракт эксперимента

Заполните для PR с экспериментом. Для инфраструктурного PR поставьте `N/A`.

| Поле | Значение |
|---|---|
| Experiment ID | `E___` |
| Run ID | `E___...` |
| Ответственный |  |
| Гипотеза |  |
| Parent experiment |  |
| Единственный изменяемый фактор |  |
| Evaluation role | `selection` / `N/A` |
| Seed |  |
| Smoke или full |  |
| ClearML task URL |  |

## Результат

| Метрика | Parent | Этот run | Delta |
|---|---:|---:|---:|
| Log loss |  |  |  |
| Accuracy |  |  |  |
| Macro F1 |  |  |  |
| ECE-15 |  |  |  |
| Brier score |  |  |  |
| A/B swap error L1 |  |  |  |
| Runtime, sec |  |  |  |

Путь к versioned result: `results/runs/<run_id>/`

Путь/ссылка и checksum больших artifacts: `artifacts/<run_id>/` или `N/A`

## Проверки

- [ ] Изменение ограничено заявленной гипотезой или инфраструктурной задачей.
- [ ] Использован frozen split; `data/splits/folds.csv` не пересоздавался.
- [ ] Для выбора модели не использованы calibration/final-holdout folds.
- [ ] `make test` проходит.
- [ ] `make check-results` проходит.
- [ ] `make collect-results` проходит и `results/leaderboard.csv` обновлен.
- [ ] Каждый новый `results/runs/<run_id>` проходит `make validate-run RUN_DIR=...`.
- [ ] В Git нет raw Kaggle data, checkpoints, токенов, `.env` и персональных путей.
- [ ] Заимствованные источники и модели добавлены в `docs/SOURCES.md`.

## Риски и ограничения

Опишите известные ограничения, неудачные проверки и что нельзя заключать из результата.
