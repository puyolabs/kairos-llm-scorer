# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres
to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.1.0] — 2026-06-15

### Performance
- **Vectorized the scoring math.** Replaced the pairwise Python cosine loop with a
  numpy `cosine_matrix` (all-pairs) and batch `closeness_weights`; adds a numpy
  dependency. Cuts the dominant CPU cost of the deterministic path.
- **Migrated the LLM screener to async.** `AnthropicScreener` now uses
  `AsyncAnthropic`; `screen` / `_supports_effort` / `score` and the eval harness are
  awaited end to end, freeing the event loop during multi-second LLM round-trips.

### Added
- **Embedding-dimension validation.** `ScoreRequest` rejects any request whose
  embedding vectors don't share one positive dimension (→ HTTP 422), and
  `SemanticTag.vector` enforces `min_length=1`.
- **Structured 422 error contract.** `/score` parses its body by hand with
  `ScoreRequest.model_validate_json`, and a custom `ValidationError` handler
  preserves FastAPI's `{"detail": [...]}` shape for malformed JSON, bad fields, and
  dimension mismatches.
- **Model-agnostic eval verdict cache + `--rekey`.** The cache is keyed by model +
  judgement + vector-stripped request + baseline (only what the screener sends
  Anthropic), so re-embedding doesn't bust it; `--rekey` losslessly migrates the
  legacy whole-request cache with no billed calls. Reports and caches are written
  per model family.

### Infrastructure
- **Reproducible builds.** Committed `uv.lock` and `.terraform.lock.hcl`; the
  Dockerfile installs with `--frozen`.
- **Standardized resource limits** at .5 vCPU / 1 GiB across Docker Compose,
  Kubernetes, and Fargate.
- **ECR `force_delete`** so `terraform destroy` no longer fails on a non-empty
  repository.
- **Explicit `--env-file .env`** in the post-receive deploy hook, fixing secret
  interpolation across included compose files.

### Tests
- Closed the v1.1.0 coverage gaps: async `/score` escalation through ASGI, empty-
  vector rejection, the async `AnthropicScreener` adapter (`screen` +
  `_supports_effort`), eval cache keying/rekey correctness, and `_iter_vectors`
  deep-nesting reach. 252 tests, up from 230.

## [1.0.0]

- Initial release: two-stage LLM job-fit scorer (deterministic baseline →
  Anthropic screener on escalation), FastAPI service, and the golden-set eval
  harness.
