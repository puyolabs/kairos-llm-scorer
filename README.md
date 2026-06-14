# kairos-llm-scorer

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)

> An LLM job-fit scoring service. Takes a job posting faced against a candidate
> profile (with embedded preferences) and returns a structured
> **apply / maybe / skip** verdict with a 0–100 match score and cited reasoning.
>
> The point of this repo is not the endpoint — it's the **eval harness**. Anyone
> can wrap an LLM in FastAPI. This service ships with a golden set, a metric, and
> a deterministic baseline to measure the model against.

---

## What it does

One job. Given:

- a **posting** (title, company, description, structured fields),
- a **profile** (the candidate's `body`, a resolved skill `ledger`, and embedded
  `preferences`: engagement type, comp floors, candidate domains, an optional hard gate),

it returns a structured verdict:

```json
{
  "decision": "apply | maybe | skip",
  "match_score": 0-100,
  "reasoning": "one paragraph citing concrete profile evidence",
  "risks_and_gaps": ["..."],
  "tailoring_hints": ["..."]
}
```

The verdict also carries deployment provenance (`version`, `build_sha`, `scorer`),
stamped by the application layer. `tailoring_hints` is populated only when
`decision = "apply"`. Scoring weighs four axes: **technical fit (40%), seniority
fit (20%), domain fit (20%), remote + timezone fit (20%)**, folded through a
multiplicative gate.

## Why it exists

It isolates an **LLM screener** as a standalone, deployable service, and pairs it
with a **deterministic arithmetic scorer** as the eval baseline — a free,
reproducible second opinion to measure the model against.

## Architecture

Hexagonal: a pure `domain`, an `application` use-case that depends only on a port,
and `infrastructure` adapters wired at the edge (`api.py`).

```
kairos-llm-scorer/
├── src/scorer/
│   ├── api.py                          # FastAPI app: GET /health, POST /score; composition root
│   ├── auth.py                         # API-key dependency for POST /score (constant-time compare)
│   ├── config.py                       # env-bound Settings (model, effort, thresholds, …)
│   ├── domain/
│   │   ├── models/                     # Pydantic contracts: Verdict, ScoreRequest, Posting, Profile, enums, mappers
│   │   └── logic/
│   │       ├── baseline.py             # deterministic four-axis scorer — the eval anchor
│   │       ├── gate.py                 # six binary gate factors → multiplicative GATE
│   │       ├── eligibility.py          # country-eligibility gating
│   │       ├── vec.py                  # cosine-similarity primitives for embedding matches
│   │       ├── tuning.py               # scorer.toml → frozen ScorerTuning, injected into the pure scorer
│   │       └── validate.py             # pre-scoring vocabulary validation
│   ├── application/
│   │   ├── score.py                    # two-stage use-case: baseline, then LLM on escalation
│   │   └── screener_port.py            # ScreenerPort Protocol — the LLM abstraction
│   └── infrastructure/
│       └── screener/
│           ├── prompt.py               # builds prompt blocks from config/screener.xml (IP); raises if absent
│           └── anthropic_adapter/screener.py # Anthropic adapter: messages.parse → Verdict
├── config/                             # provisioned config (examples committed, real files gitignored)
│   ├── screener.example.xml            #   reference screener prompt (real screener.xml is gitignored IP)
│   └── scorer.example.toml             #   reference scorer tuning (real scorer.toml is gitignored, fail-loud)
├── evals/                              # committed synthetic fixtures + the harness
│   ├── run_eval.py                     #   LLM verdict vs golden label vs deterministic baseline
│   ├── golden.jsonl                    #   35 synthetic, hand-labeled postings
│   ├── profile.json                    #   synthetic candidate profile
│   └── report.{haiku,sonnet}.md        #   decision-agreement %, score MAE, confusion matrix
└── infra/
    ├── Dockerfile                      # the image, shared by every target
    ├── docker-compose.base.yml         # the scorer service (shared via `include`)
    ├── cloud/                          # AWS Fargate (main.tf) + Kubernetes (k8s/)
    └── local/                          # self-hosted: standalone/ and cluster/ (shared edge)
```

### Three design choices worth calling out

- **Structured output via `messages.parse`.** The screener calls
  `client.messages.parse(..., output_format=Verdict)`; the SDK validates the model
  output against the pydantic `Verdict` schema — no hand-rolled JSON parsing, no
  fence-stripping. (The thinking-`effort` knob is sent only when the Models API
  reports the model supports it.)
- **Prompt caching on the profile prefix.** The profile is the long, stable part of
  every call; the posting is the short, changing part. The profile system block
  carries `cache_control: {"type": "ephemeral"}` so repeat scoring against the same
  profile reads from cache.
- **The eval harness is the load-bearing part.** A golden set of synthetic,
  hand-labeled postings; a metric (decision-agreement %, score MAE vs the labels);
  and the deterministic scorer as a baseline the LLM has to beat — so the model's
  quality is measurable, not assumed.

### Two-stage scoring

`POST /score` runs the deterministic baseline first (free, no network), then
escalates to the LLM screener only when the request's `sonnet_judgement` mode says
to: `wide` (escalate `maybe`/`apply` plus near-miss `skip`s above `escalate_floor`),
`narrow` (escalate `apply` only — the tailoring pass), or `off` (return the
arithmetic verdict as-is). Default is `wide`.

## Quickstart

```bash
uv sync
export ANTHROPIC_API_KEY=sk-ant-...

# Run the service
uv run uvicorn scorer.api:app --reload

# Health check (always open)
curl localhost:8000/health

# Score one posting. The POST body is a ScoreRequest: { posting, profile, sonnet_judgement? }
# where preferences live inside profile.
# Locally /score is open (no KAIROS_SCORER_API_KEYS set). Once keys are configured,
# present one:  -H 'Authorization: Bearer <key>'   or   -H 'X-API-Key: <key>'

# Run the eval harness against the golden set (baseline-only by default)
uv run python -m evals.run_eval                  # --judgement off  (no network)
uv run python -m evals.run_eval --judgement wide # real, billed LLM calls
uv run python -m evals.run_eval --limit 10       # first N examples only
```

The LLM stage runs only when both `ANTHROPIC_API_KEY` and the (gitignored, IP)
`screener.xml` are present; otherwise the harness reports the baseline alone.

## Eval results

Against the committed golden set (35 synthetic postings — apply=12, maybe=10,
skip=13 — faced against a synthetic profile; the baseline escalated 26 to the LLM).
Full reports: [`evals/report.sonnet.md`](evals/report.sonnet.md),
[`evals/report.haiku.md`](evals/report.haiku.md).

| Metric                       | Sonnet 4.6 | Haiku 4.5 | Deterministic baseline |
|------------------------------|------------|-----------|------------------------|
| Decision agreement vs golden | 97.1%      | 85.7%     | 65.7%                  |
| Score MAE vs golden          | 6.5        | 7.5       | 15.7                   |

## Deployment

One image, split into `infra/cloud/` and `infra/local/` — see
[`infra/README.md`](infra/README.md) for full commands:

- **local / standalone** — a dedicated VPS / laptop; its own Caddy owns :80/:443
  (`infra/local/standalone/`). Run with
  `docker compose -f infra/local/standalone/docker-compose.yml up -d --wait`.
- **local / cluster** — behind a shared edge proxy (`infra/local/cluster/` + a
  `post-receive` hook).
- **cloud** — AWS Fargate via Terraform (`infra/cloud/main.tf`, ECR + Fargate
  behind an ALB) or any Kubernetes cluster (`infra/cloud/k8s/deployment.yaml`).

Each local topology's compose file `include`s `infra/docker-compose.base.yml` for
the scorer service and adds its own Caddy. The container listens on `:8000` and
serves an unauthenticated `GET /health` (also the container healthcheck probe).

## Auth

Built to face the internet: `POST /score` is gated by an app-level API key, so
authorized and unauthorized callers are separated in the application itself, not
just at the edge.

- `KAIROS_SCORER_API_KEYS` — comma-separated accepted keys. Callers present one via
  `Authorization: Bearer <key>` or `X-API-Key`; a missing/invalid key is a `401`
  (constant-time compared).
- Empty (the default) **disables** the gate — the right posture for localhost and
  the LAN surface. Set it before unblocking any public surface, and the gate
  protects `/score` immediately. `GET /health` is always open.

The full secret wiring (compose, Terraform Secrets Manager, k8s Secret) is in
`infra/`. The gate is authentication; per-key rate limiting is a natural next step.

## Model

Defaults to `claude-sonnet-4-6` for the screener call. The model id is configurable
via `KAIROS_SCORER_MODEL`; thinking depth via `KAIROS_SCORER_EFFORT`
(`low|medium|high|max`) and output cap via `KAIROS_SCORER_MAX_TOKENS`. Structured
output and prompt caching are model-agnostic across the current Claude family.

## Screener prompt (IP)

The real `config/screener.xml` is the proprietary prompt and is gitignored. The
loader **raises** if it is absent (fail-loud) — there is no silent fallback to the
committed `config/screener.example.xml`, which is a reference template only.
`/health` and non-escalating scores work without it (only escalation to the LLM
needs it).

## Scorer tuning

The deterministic scorer's numeric knobs (axis weights, tier credit, thresholds,
region offsets) live in `config/scorer.toml`, loaded at the boundary and injected
into the pure scorer (the domain reads no files). It is gitignored and **fail-loud**:
the loader **raises** if it is absent — no fallback to the committed
`config/scorer.example.toml` reference. Because it is fail-loud, the baseline
itself requires `config/scorer.toml`, so **`/score` and the eval both need it
provisioned**; `/health` needs neither it nor the screener prompt. The axis weights
here are also the single source the LLM prompt's rubric is interpolated from, so
the prompt cannot drift from the scorer.

## License

Licensed under the [Apache License, Version 2.0](LICENSE). See [`NOTICE`](NOTICE)
for attribution.
