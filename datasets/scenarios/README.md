# Scenario datasets

Benchmark and adversarial scenarios are defined **in code** for determinism:
see `app/evaluation/scenarios.py` (10 benchmark scenarios) and
`app/evaluation/adversarial.py` (5 safety scenarios). Each scenario derives its
input at runtime from the healthy baseline fixtures in `datasets/fixtures/`,
so no duplicated data files are needed here.
