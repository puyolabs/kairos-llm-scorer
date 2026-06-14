# Deployment

One image, two worlds. The container listens on **:8000**, serves an
unauthenticated `GET /health`, and gates `POST /score` behind an API key once
`KAIROS_SCORER_API_KEYS` is set (see [Auth](#auth)).

```
infra/
├── Dockerfile                     # the image, shared by every target
├── Dockerfile.dockerignore        # build-context excludes (BuildKit)
├── docker-compose.base.yml        # the scorer service, shared via `include`
├── cloud/                         # managed cloud
│   ├── main.tf                    #   AWS: ECR + Fargate + ALB
│   └── k8s/deployment.yaml        #   any Kubernetes cluster
└── local/                         # self-hosted (docker compose)
    ├── standalone/                #   this host owns :80/:443 (own Caddy)
    │   ├── docker-compose.yml
    │   └── Caddyfile
    └── cluster/                   #   shared host behind a shared edge proxy
        ├── docker-compose.yml
        ├── Caddyfile
        ├── post-receive.sh        #   git-push deploy hook
        └── edge/kairos-llm-scorer.caddy
```

Each `local/<topology>/docker-compose.yml` is self-contained — it `include`s
`../../docker-compose.base.yml` for the scorer service and adds its own Caddy.
Run one with `docker compose -f infra/local/<topology>/docker-compose.yml …`.

## Auth

The service may face the internet, so authorized vs unauthorized callers are
separated **in the app**, not just at the edge:

- `KAIROS_SCORER_API_KEYS` empty → gate **off** (localhost / LAN).
- non-empty (comma-separated) → `POST /score` requires a matching key via
  `Authorization: Bearer <key>` or `X-API-Key`; otherwise **401**. `/health` stays open.

Set it before any public surface is unblocked. The shared edge's `public_gate` is
a separate, coarse all-or-nothing block; this is the per-call check.

## Provisioned config (IP / tuning)

Both live under `config/` and are gitignored; the committed `.example` files are
references only. They are **not baked into the image** — `Dockerfile.dockerignore`
keeps the real files out of the build context (only the `.example.*` get copied).
Instead the scorer service bind-mounts the deploy workdir's `config/` over
`/app/config` read-only (`../config:/app/config:ro` in `docker-compose.base.yml`;
`../config` resolves from `infra/` to the repo-root / workdir `config/`). Drop the
real files into `config/` once (`checkout -f` and rebuilds preserve the untracked
files); they reach the container via the mount, durably across `down`/`up` and
redeploys.

- `config/screener.xml` — the real screener prompt (IP). The loader **raises** if it
  is absent (fail-loud), so any deploy that escalates to the LLM must provision it.
- `config/scorer.toml` — the deterministic scorer's tuning. Also **fail-loud**, and
  the baseline needs it, so `/score` (and the eval) require it regardless of the
  LLM. `/health` needs neither.

## local / standalone

Its own Caddy owns the host's :80/:443.

```bash
cp .env.example .env    # set ANTHROPIC_API_KEY, KAIROS_SCORER_API_KEYS, SCORER_SITE_ADDRESS
docker compose -f infra/local/standalone/docker-compose.yml up -d --wait
curl -H 'X-API-Key: clientkey1' -H 'Content-Type: application/json' -X POST localhost/score \
  -d '{"posting":{"kind":"job","source_id":"x","external_id":"1","canonical_key":"x::1","url":"","title":"Staff Engineer","company":"Example","description":"Remote senior IC","posted_at":"","fetched_at":"","location_text":"Remote","remote":"yes","seniority_hint":"staff"},"profile":{"body":"Senior IC, remote, TypeScript/Postgres."}}'
```

Set `SCORER_SITE_ADDRESS=scorer.example.com` (a bare domain) for automatic HTTPS.

## local / cluster

Mirrors kairos: push to a bare repo, a `post-receive` hook builds + brings up the
stack and registers the edge route. The shared edge proxy (`~/edge`) owns :80/:443 and
routes `scorer.cluster.lan` (LAN) and `scorer.example.com` (public, via an
outbound tunnel) to this stack's inner Caddy alias `kairos-llm-scorer-inner`.

**One-time host setup**

```bash
ssh cluster-host
git init --bare ~/kairos-llm-scorer.git
docker network create edge          # no-op if another project already made it
# install the hook:
#   scp infra/local/cluster/post-receive.sh cluster-host:~/kairos-llm-scorer.git/hooks/post-receive
chmod +x ~/kairos-llm-scorer.git/hooks/post-receive

# Working tree: created on first deploy. Drop these in once (survive checkout -f):
mkdir -p ~/kairos-llm-scorer
#  - .env  with: ANTHROPIC_API_KEY=sk-ant-...
#                KAIROS_SCORER_API_KEYS=clientkey1,clientkey2
#  - config/screener.xml  (the IP prompt)
#  - config/scorer.toml   (the deterministic scorer tuning; required by the baseline)
```

**Deploy** (from a workstation):

```bash
git remote add cluster cluster-host:kairos-llm-scorer.git   # once
git push cluster stg   # the branch this host tracks (SCORER_DEPLOY_BRANCH: prd|stg|lab)
```

The hook builds, waits on `/health`, copies
`infra/local/cluster/edge/kairos-llm-scorer.caddy` into `~/edge/conf.d/`, and
reloads the edge. Going public is an edge concern: bring up its tunnel daemon
(`public` profile) and keep `scorer.example.com` out of `EDGE_PUBLIC_BLOCK`.
Do that **only after** `KAIROS_SCORER_API_KEYS` is set.

## cloud / AWS Fargate (`cloud/main.tf`)

ECR + ECS Fargate behind an internet-facing ALB, in the default VPC. Secrets for
`ANTHROPIC_API_KEY` and `KAIROS_SCORER_API_KEYS` live in Secrets Manager and are
injected into the task.

```bash
cd infra/cloud
terraform init
terraform apply                       # creates ECR + infra (tasks fail health until an image exists)

ECR=$(terraform output -raw ecr_repository_url)
REGION=us-east-1; ACCOUNT=$(aws sts get-caller-identity --query Account --output text)
aws ecr get-login-password --region "$REGION" \
  | docker login --username AWS --password-stdin "${ACCOUNT}.dkr.ecr.${REGION}.amazonaws.com"
# Build context is the repo root; the Dockerfile lives in infra/:
docker build -f ../Dockerfile -t "$ECR:latest" ../..
docker push "$ECR:latest"

# Secrets (out-of-band; kept out of TF state):
aws secretsmanager put-secret-value --secret-id kairos-llm-scorer/anthropic-api-key --secret-string 'sk-ant-...'
aws secretsmanager put-secret-value --secret-id kairos-llm-scorer/scorer-api-keys   --secret-string 'clientkey1,clientkey2'

aws ecs update-service --cluster kairos-llm-scorer --service kairos-llm-scorer --force-new-deployment
terraform output -raw alb_dns_name
```

Or set both at apply: `TF_VAR_anthropic_api_key=… TF_VAR_scorer_api_keys=… terraform apply`.

## cloud / Kubernetes (`cloud/k8s/`)

```bash
kubectl create secret generic kairos-llm-scorer \
  --from-literal=anthropic-api-key=sk-ant-... \
  --from-literal=scorer-api-keys='clientkey1,clientkey2'
# edit the image + Ingress host in infra/cloud/k8s/deployment.yaml, then:
kubectl apply -f infra/cloud/k8s/deployment.yaml
```
