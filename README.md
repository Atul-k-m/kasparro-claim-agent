# Kasparro FSE Intern Assessment: claim extraction and classification agent

A Python agent reads a long AI answer about speakers. It extracts every factual claim sentence with exact character spans and labels each one `supported`, `contradicted` or `unknown`. The labels come from `brand_facts.csv`, fetched through live model-requested `get_brand_facts` tool calls. The project also includes a deterministic harness, a FastAPI service with PostgreSQL storage, and controlled tests.

## Status at submission time (honest)

| Item | State |
|---|---|
| Agent (tool use, extraction, Python span location, validation, 3-turn cap, timeout) | Done |
| Deterministic harness (exact spans, extraction and joint P/R/F1, missed, extra and wrong-label records, error accounting) | Done |
| FastAPI `POST /classify`, `POST /evaluation-runs`, `GET /evaluation-runs/{id}` + PostgreSQL | Done |
| Controlled tests | 52 passing (see `reports/test_output.txt`) |
| v2 targeted change (`app/prompts/v2.md`) | Done: one added extraction rule (see Investigation) |
| Live run: v1 development | **In progress at the deadline** (Groq free tier: 8k tokens/min, about 40 s per answer). Report written to `reports/` when complete |
| v2 development, v1 check, v2 check | **Queued** to run one after another once v1 development finishes; reports will be added to `reports/` |

## Layout

```
app/
  config.py      settings, versions, provider registry (no secrets)
  data.py        CSV loading (csv module, newline='', UTF-8, typed values, strict booleans)
  tools.py       FactStore + get_brand_facts tool spec + validated executor
  provider.py    OpenAI-compatible adapter (Groq/OpenAI/Gemini) + FakeProvider for tests
  agent.py       3-turn tool-calling loop; quotes from model, offsets from Python
  validate.py    JSON parsing, quote location, span/label/fact-id/overlap validation
  scorer.py      exact-span scorer and metrics
  runner.py      sequential evaluation runs; reports + PostgreSQL persistence
  db.py          schema (runs, answer_results, claim_results) + parameterized SQL
  api.py         thin FastAPI routes
  prompts/v1.md  baseline prompt (frozen)
scripts/run_eval.py   CLI for the four runs
scripts/start_db.py   local embedded PostgreSQL (optional)
tests/                controlled tests (fake provider; real PostgreSQL test database)
reports/              live-run reports (.json + .md), per-answer traces, console logs
```

## Setup

Requires Python 3.11+.

```bash
python -m pip install -r requirements.txt
cp .env.example .env          # fill in GROQ_API_KEY (or OPENAI_API_KEY / GEMINI_API_KEY + LLM_PROVIDER)
python scripts/start_db.py    # optional: starts a local PostgreSQL, prints DATABASE_URL / TEST_DATABASE_URL
```

Put the printed `DATABASE_URL` into `.env`. Any PostgreSQL 14+ works, for example `postgresql://user:pass@localhost:5432/kasparro`. The schema is created automatically.

## Run

```bash
python scripts/run_eval.py --version v1 --split development
python scripts/run_eval.py --version v2 --split development
python scripts/run_eval.py --version v1 --split check
python scripts/run_eval.py --version v2 --split check
uvicorn app.api:app_factory --factory --port 8000
python -m pytest -v
```

Each run writes `reports/<version>_<split>_<timestamp>.json` and a `.md` summary. The JSON holds the run metadata, settings, file hashes, counts, metrics, missed, extra and wrong-label records, and every answer outcome with its predictions. Per-answer model/tool exchanges go to `reports/traces/<run_id>/<answer_id>.json`. The same data is stored in PostgreSQL.

## Design

**Agent contract**
- The input is exactly `answer_id, brand_id, question, ai_answer`. Gold data is never loaded on the inference path; a test checks every payload for this.
- Turn 1 uses `tool_choice="required"`, so the model must request `get_brand_facts`. It can request several brands in parallel, for example for cross-brand comparisons. Later turns use `tool_choice="none"` and return JSON only. The hard cap is 3 model turns.
- The executor accepts only the tool name `get_brand_facts` and exactly one `brand_id` string argument that exists in the catalogue. Anything else is an `illegal_tool`, `invalid_tool_arguments` or `unknown_brand` answer error. Text from the answer is never executed.
- The model returns **quotes**, never offsets. Python finds each quote verbatim in the exact decoded `ai_answer`. A repeated sentence is resolved by order: the first free occurrence after the previous claim. Python then builds `claim_id = answer_id:start:end` and validates bounds, quote equality, whitespace, unique IDs, no overlaps, labels, fact IDs and a reason of at most two sentences.
- Any failure (provider error, timeout, invalid output, illegal tool, turn exhaustion) becomes an answer-level error with no accepted predictions. It is never converted to `unknown`.
- Provider request timeout: **90 s** (`LLM_TIMEOUT_S`). The SDK's own retries are off. HTTP 429 is retried up to 6 times using `retry-after`. A daily-quota 429 interrupts the run, which is marked failed with partial evidence kept.

**Harness**
- A prediction matches only when `(answer_id, start_char, end_char)` is identical, after quote equality is checked. Each gold span matches at most once.
- Extraction P/R/F1 = M/P, M/G, 2M/(P+G). Joint P/R/F1 = C/P, C/G, 2C/(P+G). Coverage = C/G.
- Answers that error keep all their gold claims in G as missed. Missing or duplicated answer outcomes are rejected.
- Check runs keep G, M, C, matches and all quality metrics `null`.

**Database**
- `runs` holds versions, provider, model, settings, source revision (git SHA or code hash), CSV hashes, counts, metrics, and missed/extra ID lists in JSONB.
- `answer_results` has `PRIMARY KEY (run_id, answer_id)` and stores status, sanitized error, latency and trace_ref.
- `claim_results` has `PRIMARY KEY (run_id, answer_id, start_char, end_char)`, a foreign key to `answer_results`, label CHECK constraints and optional gold match columns.
- All SQL is parameterized, and each answer and its claims are written in one transaction.

**API errors** are returned as `{"error": {"type", "message"}}` and never echo request values.

| Status | Cases |
|---|---|
| 422 | invalid request, or extra fields such as `claims`, `gold` or labels; invalid UUID |
| 404 | unknown primary brand; missing run |
| 502 | provider error or invalid output |
| 504 | timeout |

## Versions and inputs

- Dataset `long-brand-answers-50-v4`, facts `speaker-catalogue-v4`, evaluator `sentence-spans-and-labels-v4`.
- Provider **Groq** (OpenAI-compatible API), model `openai/gpt-oss-120b`, temperature 0, `reasoning_effort=low`, max output 3500 tokens.
- **Deviation:** the brief lists OpenAI, Anthropic or Gemini. Groq was used because it was the key available. Switching is a `.env` change only (`LLM_PROVIDER=openai|gemini`, `LLM_MODEL=...`).

Issued CSVs are kept unmodified in the repository root:

| File | SHA-256 |
|---|---|
| answers.csv | fa28e813ea54b4e5ea985585f22f084d826241e1182accbc27ace17066125724 |
| brand_facts.csv | 2190d40fad2dd167ef2437956c43d595aaf2554ea673a4789b9343cb7e2936f2 |
| development_gold.csv | afe84cc78f304afb6a4aaa505dd7504af80fcbf8bc507806aa5ff0aab6bf4850 |

## Investigation so far

**Aborted pre-baseline run.** The first v1 development attempt used `tool_choice="auto"` on turn 2. gpt-oss on Groq then tried to send its final JSON as a call to an undeclared tool named `json`, and the provider rejected it with HTTP 400 `tool_use_failed`. This happened on 3 of 3 answers. I stopped the run to save the daily token quota and kept its evidence in `reports/aborted_pre_baseline/`; the database row is marked `failed`. The executor was changed before the v1 baseline: turn 1 requires the tool and later turns are answer-only. Provider `tool_use_failed` is now classified as `illegal_tool`.

**v1 error diagnosis (first 11 development answers, A01 to A11, scored by the harness).**

| Answers | G | P | M | C | Joint F1 | Extra predictions |
|---|---|---|---|---|---|---|
| 11 | 104 | 100 | 100 | 98 | 0.961 | 0 |

- **Missed (4):** all in A10, and all have gold label `unknown`. They cover manufacturing origin, a future price drop, a return/refund policy and multipoint Bluetooth. The trace for A10 shows the model returned only 6 claims, every one `supported` or `contradicted`. It treated "no matching fact" as "not a claim" and dropped those sentences.
- **Wrong label (2):** A02 is a below-freezing battery runtime, labeled `supported` where gold is `unknown` (condition scope). A09 is a reversed price-difference direction, labeled `supported` where gold is `contradicted`.

**v2 targeted change.** v2 is v1 plus one extraction rule; `diff app/prompts/v1.md app/prompts/v2.md` shows a single added line. The rule says a factual sentence still counts when the catalogue has no matching fact: extract it, label it `unknown` and never omit it, with examples such as origin, future prices or promises, return terms, connection features and other unlisted specifications. It targets the largest error group, missed `unknown` claims. The expected risk is more extra predictions, if advice sentences start being extracted as `unknown`; the v2 development run measures this. The two labeling errors are deliberately left unchanged so that only one behavior changes.

The diagnosis used v1's first 11 answers because of the deadline. The full v1 development report will be added when the run completes. The data, evaluator, provider, model and settings are identical for v1 and v2. I have not tuned on the check answers.

## Unfinished work

- Complete the v1 development run (in progress) and the v2 development run, then run v1 check and v2 check with both versions frozen. These are queued to run automatically one after another.
- After the runs: compare v1 and v2 on all 40 development answers and report the measured effect, including any regressions or no gain.
- The cause is throughput: the Groq free tier allows 8,000 tokens per minute, and one answer uses about 5.5k tokens over 2 turns. The 100 answer-runs need about 1.5 hours of wall time. A paid tier or another provider would finish in about 15 minutes.

## Time spent

About 1 hour of active build time with AI assistance (see below). A power cut before I could start reduced the time I had for the assessment. Within the time that remained, the unfinished live runs were limited by the Groq free-tier throughput described above.

## AI assistance disclosure

Built with Claude Code (Anthropic). Claude Code wrote most of the code from the assessment brief. The log in [docs/AI_ASSISTANCE.md](docs/AI_ASSISTANCE.md) shows what I asked for and what I checked myself.
