# Self-Healing Data Pipeline Agent

A production-minded autonomous agent that detects failures and schema drift in
ETL/ELT pipelines, investigates the root cause, proposes a safe transformation
fix, validates it against shadow data, executes it through a controlled rollout,
and automatically rolls back when validation or post-deployment health checks
fail.

> **Core principle:** the agent never blindly mutates production data.
> It observes → diagnoses → proposes → validates → stages → gets approval →
> executes a canary → monitors → rolls back on any health degradation.

This is not a chatbot that explains pipeline errors. It is an autonomous system
with a bounded state machine, a policy engine that gates every action, and an
evaluation framework whose headline metric — **data loss rate** — is held at
**zero** across all benchmark and adversarial scenarios.

---

## The problem it solves

A source system quietly changes a column type:

```json
{"user_id": 123, "amount": 49.99, "currency": "USD"}   // yesterday
{"user_id": 123, "amount": "49.99", "currency": "USD"} // today
```

The pipeline expects `amount: DOUBLE` and now receives `VARCHAR`. The agent:

1. **Detects** the drift deterministically (schema comparison, quality checks,
   statistical anomaly detection — no LLM needed for measurement).
2. **Diagnoses** the root cause and inspects downstream lineage to compute the
   blast radius (`amount` → `clean.orders` → `daily_revenue`).
3. **Proposes** a declarative, constrained fix: `cast_type(amount, DOUBLE)`.
4. **Shadow-validates** the candidate against the input baseline: schema,
   row preservation, null rates, business metrics (`revenue <- amount`).
5. **Asks the policy engine** — never itself — whether it may proceed
   (LOW risk auto-stages; HIGH risk is rejected without an explicit gate).
6. **Stages** a new pipeline version and runs a **95/5 canary**.
7. **Monitors** health and **rolls back** deterministically if anything degrades.

---

## Architecture

```text
        Source (CSV fixtures)
              │
        Ingestion (load_source)
              │
        Raw observed data ─────────────┐
              │                        │
   ┌──────────▼──────────┐    ┌────────▼─────────┐
   │  Pipeline Runner    │    │  Agent Loop       │
   │  (typed stages)     │◄───┤  bounded FSM      │
   │  clean → aggregate  │    │  OBSERVE…ROLLBACK │
   └──────────┬──────────┘    └────────┬─────────┘
              │                        │ tools (policy-gated)
       ┌──────▼──────┐          ┌──────▼───────┐
       │  Warehouse  │          │ Metadata     │
       │  (versioned)│          │ store        │
       └─────────────┘          └──────────────┘
        main / shadow /          state, schema registry,
        canary namespaces        metric history, rollbacks
```

The agent loop is an **explicit, bounded state machine**:

```text
OBSERVE → DETECT → DIAGNOSE → PLAN → PROPOSE → VALIDATE → APPROVAL
    → STAGE → CANARY → MONITOR → SUCCESS
                                    │
                                    └──→ ROLLBACK   (deterministic, idempotent)
```

Any guardrail breach (iteration budget, tool-call budget, execution timeout,
resource budget) ends the run in `SAFE_STOP` — the agent prefers doing nothing
over making an unsafe change.

---

## Safety model

| Layer | Mechanism |
|---|---|
| **Policy engine** | Every tool carries an action class (`READ_ONLY`, `SHADOW_WRITE`, `STAGING_WRITE`, `PRODUCTION_WRITE`, `ROLLBACK`). Tools are gated per phase; the LLM never decides permissions. |
| **Constrained fixes** | The LLM can only emit declarative operations (`cast_type`, `fill_null`, `deduplicate`, `parse_timestamp`, …). Arbitrary SQL and shell are impossible by construction; unknown operations are dropped. |
| **Risk-based approval** | `LOW` (safe coercion) auto-stages · `MEDIUM` (rename, dedup) requires explicit approval · `HIGH` (row deletion) is rejected without a gate. |
| **Shadow validation** | Candidates run against an isolated namespace and must match the expected output schema, preserve rows, and keep business metrics within 5% of the input baseline (including aggregated aliases like `revenue <- amount`). |
| **Canary** | 95% current / 5% candidate partition; candidate failure or quality violation ⇒ `CANARY_FAILED`. |
| **Rollback** | Versioned warehouse snapshots; restore is deterministic and idempotent, with an audited reason. |

---

## Quickstart

```bash
# 1. Install (Python 3.12+)
pip install -e ".[dev]"

# 2. Generate the demo fixtures
python datasets/fixtures/_generate.py

# 3. Run the full test suite (unit / integration / agent / safety / evaluation)
make test

# 4. Run the evaluation: 10 benchmark + 5 adversarial scenarios
make evals            # or: python -m app.evaluation.runner --scenarios all
# Exit code is non-zero unless data_loss_rate == 0.

# 5. Serve the API
make api              # uvicorn app.api.app:app on :8000
```

### API

```bash
curl localhost:8000/health
curl localhost:8000/pipelines
curl -X POST localhost:8000/pipelines/orders/triage       # healthy source -> no action
curl localhost:8000/runs/<run_id>                          # full persisted triage state
```

`POST .../triage` accepts `{"source": "<path-or-fixture-key>", "limit": 1000}`.
Runs are bounded (sub-second on demo-scale data) and state is persisted for
later inspection.

---

## Evaluation

`python -m app.evaluation.runner --scenarios all` executes **10 benchmark
scenarios** (type drift, column addition, rename hypothesis, timestamp format,
null spike, duplicate spike, invalid numerics, volume anomaly, referential
integrity break, transformation regression) and **5 adversarial safety
scenarios** (prompt injection treated as data, destructive proposal, mass
deletion, silent corruption, partial-deployment failure).

Latest verified results (deterministic mode, zero API cost):

| Metric | Value |
|---|---|
| Outcome match rate | **1.000** (15/15) |
| Drift detection rate | 0.933 |
| Root-cause accuracy | 0.800 |
| Fix success rate | 0.333 |
| Validation pass rate | 0.600 |
| Rollback success rate | 0.067 |
| **Data loss rate** | **0.000** |
| False repair rate | **0.000** |
| Unnecessary change rate | **0.000** |
| Avg tool calls / latency | 2.07 / 6 ms |
| Estimated cost | $0.00 |

Interpretation: the low fix/validation rates are by design. Most scenarios are
*investigate* or *blocked* cases (rename hypothesis, volume anomaly, duplicate
spike, destructive proposals) where the correct behaviour is to surface evidence
and **not** mutate. The safety-critical numbers are the three zeros: no data
loss, no repairs without a correct diagnosis, no changes where none were
warranted. Reports are written to `reports/evaluation.md` and
`reports/evaluation.json`.

---

## Repository structure

```text
app/
├── agent/            orchestrator (bounded FSM), policies, diagnosis, planner,
│                     LLM providers (deterministic + openai), prompts
├── api/              minimal FastAPI surface
├── data/             profiler, schema comparison, quality, anomaly, lineage, loader
├── evaluation/       scenarios, adversarial, runner, metrics, reports
├── models/           pydantic domain types (schemas, drift, fixes, agent state)
├── observability/    metric repository + health monitor
├── pipeline/         typed stages, transformation registry, runner, registry
├── rollback/         deterministic idempotent version restore
├── storage/          protocols + SQLite and in-memory backends
├── tools/            typed, safety-classified tool registry
├── validation/       shadow validation engine
└── config.py         env-driven settings (SHP_* prefix)

datasets/fixtures/    reproducible healthy baseline (orders, customers)
pipelines/            typed pipeline definitions (JSON)
tests/                unit / integration / agent / safety / evaluation
```

---

## Configuration

All settings are environment-driven (prefix `SHP_`, see `.env.example`):

| Variable | Default | Purpose |
|---|---|---|
| `SHP_LLM_PROVIDER` | `deterministic` | `deterministic` (offline) or `openai` |
| `OPENAI_API_KEY` / `OPENAI_MODEL` | — | only used by the OpenAI provider |
| `SHP_MAX_ITERATIONS` | `12` | agent cycle budget |
| `SHP_MAX_TOOL_CALLS` | `40` | attempt budget |
| `SHP_EXECUTION_TIMEOUT_SECONDS` | `300` | hard execution timeout |
| `SHP_MAX_BUDGET_ROWS` | `2000000` | resource guardrail |
| `SHP_APPROVAL_MODE` | `auto` | `auto` / `manual` / `human_in_loop` |
| `SHP_STORAGE_BACKEND` | `sqlite` | `sqlite` or `memory` |
| `SHP_DB_PATH` | `./data/pipeline.db` | SQLite metadata + warehouse file |

---

## Design decisions

- **Deterministic first.** Basic statistics, drift detection and anomaly
  detection are computed deterministically; the LLM only turns evidence into a
  structured proposal. The default `DeterministicLLM` makes every scenario fully
  reproducible offline; `OpenAIProvider` implements the same protocol for live
  use (unavailable without a key, and its output is sanitized through the same
  constrained-operation parser).
- **SQLite + in-memory behind protocols.** `MetadataStore` and `Warehouse` are
  protocols, so DuckDB/Postgres adapters are drop-in replacements rather than a
  rewrite. Warehouse payloads are stored as Parquet blobs so dtypes survive the
  round trip — snapshot/restore stays deterministic.
- **No orchestrator dependency.** The pipeline runner is a lightweight typed
  stage executor; Airflow/Kafka/etc. would be adapters, not foundations.

## Extending

- **Real LLM:** set `SHP_LLM_PROVIDER=openai` and `OPENAI_API_KEY`. The provider
  is mock-tested; the policy engine still gates everything it emits.
- **New storage:** implement `MetadataStore`/`Warehouse` (e.g. DuckDB, Postgres)
  and register it in `app/storage/__init__.py`.
- **New scenarios:** add a `Scenario` to `app/evaluation/scenarios.py` with a
  `build` function, expected diagnosis, acceptable fix, and expected outcome.
