# Deploying Truth Shield

The app is one Docker image: a multi-stage build compiles the Svelte bundle,
then FastAPI serves both that bundle and the API. **One URL is the whole
product** — no CORS wiring, no second host to keep alive.

```
┌─────────────────── truthshield image ───────────────────┐
│  stage 1  node:22   npm ci && vite build  →  dist/      │
│  stage 2  python    torch (CPU) + transformers          │
│                     weights baked in at build time      │
│                     uvicorn main:app  :7860             │
│                       ├─ /            → Svelte SPA      │
│                       ├─ /analyze     → RoBERTa         │
│                       ├─ /analyze_stream → NDJSON stream│
│                       └─ /health      → readiness       │
└─────────────────────────────────────────────────────────┘
```

## What the deployment needs

Measured on a 512-token input (CPU, fp32):

| Stage | RSS |
|---|---|
| torch imported | 182 MB |
| both checkpoints + GPT-2 loaded | 411 MB |
| after `/analyze` phase 1 | 846 MB |
| after `/analyze` phase 2 | **1626 MB** |

| Requirement | Value | Why |
|---|---|---|
| RAM | **2 GB** | peak is GPT-2 plus the 12-layer attention tensors the rollout needs |
| Disk | **~6 GB** | torch CPU wheel, transformers, and both checkpoints |
| CPU | 2 vCPU is comfortable | inference is CPU-only; `/analyze` lands around 1–3 s |
| GPU | not required | `main.py` picks CUDA only if it is there |

That 1.6 GB peak rules out every free 512 MB tier — Render Free and Koyeb Free
both OOM on model load, and Vercel/Netlify functions cap the bundle at 250 MB
before RAM is even a question.

## Recommended: Modal

Modal's Starter plan is $0/month plus usage with **$30/month of credits**, but a
payment method must be on file to use the account — without one you get a small
starter grant (~$1), which is enough to demo but not to leave running.

Two billing facts drive the configuration in `modal_app.py`:

- Modal charges **`max(request, actual usage)`**, and containers may burst well
  above the request. Torch sizes its thread pool to every visible core, so an
  uncapped container is a billing leak. `OMP_NUM_THREADS` and a hard `cpu` limit
  pin it.
- **Idle time dominates** on a low-traffic demo. A container kept warm after a
  request bills at the CPU *request* the whole time, so the request is 1 core and
  `scaledown_window` is 60 s.

| | Rate | Idle (1 core) | Active (2 cores) |
|---|---|---|---|
| CPU | $0.0000131 / core / s | $0.0000131 | $0.0000262 |
| Memory | $0.00000222 / GiB / s | $0.0000044 | $0.0000044 |
| | **total / s** | **$0.0000175** | **$0.0000306** |

That is **$0.063 per idle hour**. A visitor session — a few seconds of inference
plus the 60 s warm window — costs about **$0.0011**, so even the ~$1 starter
grant covers roughly **850 sessions**, and the full $30 covers ~25,000.

Raising `scaledown_window` back to 300 s costs about 8x more per visitor for a
faster second request; worth it only under real traffic.

Cold starts are handled by `enable_memory_snapshot=True`: Modal snapshots the
container after the weights are in memory and restores that image on later
boots, so a cold visitor waits seconds rather than the ~40 s a fresh load takes.

```bash
pip install modal
modal setup                                   # one-time browser auth

cd frontend && npm ci && npm run build && cd ..   # modal_app.py bakes in dist/

modal secret create truth-shield-keys     NVIDIA_API_KEY=... GNEWS_API_KEY=...

modal deploy modal_app.py
```

The deploy prints the URL — `https://<workspace>--truth-shield-truthshield-web.modal.run`.
Functional but not pretty; pair it with the Vercel frontend below if you want a
clean link on a portfolio.

## Hugging Face Spaces (requires PRO as of 2026)

CPU Basic gives 2 vCPU / 16 GB RAM / 50 GB disk and builds Dockerfiles
natively. **Docker Spaces are no longer free** — `POST /api/repos/create`
returns HTTP 402:

> Static Spaces are free for everyone, but hosting Gradio and Docker Spaces on
> free cpu-basic requires a PRO subscription.

With PRO ($9/mo) this is the least-effort option; everything here targets it.

```powershell
.\deploy\deploy_hf_space.ps1 -User <your-hf-username> -Token hf_xxx
```

The script creates the Space, pushes the source, and prints the URL. First
build runs 10–15 minutes (torch + weights); later pushes reuse cached layers.

Then set the two secrets under **Settings → Variables and secrets**:

| Secret | Purpose | Free key |
|---|---|---|
| `NVIDIA_API_KEY` | reasoning agent | <https://build.nvidia.com> |
| `GNEWS_API_KEY` | live web verification | <https://gnews.io> (100 req/day) |

Without them the Space still works — it degrades to local model scoring and
says so in the UI.

**Result:** `https://<user>-truth-shield.hf.space`

### Sleep behaviour

A free Space sleeps after 48 h idle and takes ~40 s to wake. For a portfolio
link that is usually fine. To remove it: Settings → upgrade hardware, or ping
`/health` on a schedule.

## Optional: nicer URL via Vercel

Keep the Space as the API and put the UI on Vercel for a CDN-backed link and a
custom domain.

1. Import the GitHub repo at <https://vercel.com/new>. The root `vercel.json`
   already sets the build command, output directory, and SPA rewrites.
2. Add environment variable `VITE_API_URL=https://<user>-truth-shield.hf.space`.
3. Back on the Space, set `ALLOWED_ORIGINS` to your Vercel URL to tighten CORS
   (it defaults to `*`).

**Result:** `https://truth-shield.vercel.app` calling the Space for inference.

## Other hosts

Any Docker PaaS works — the image reads `$PORT`.

| Host | Notes |
|---|---|
| Render | Needs **Standard** ($25/mo) for 2 GB. Free and Starter are both 512 MB and OOM. |
| Railway | Detects the Dockerfile; usage-based after the trial credit. |
| Fly.io | `fly launch` then `fly scale memory 2048`. |
| Google Cloud Run | Best $0 option. Free tier covers 360k GiB-s/month, so scale-to-zero demo traffic costs nothing. Set `--memory=2Gi`. Cold start is ~30-60 s while weights load; `--min-instances=1` removes it but costs ~$12/mo. Billing account required. |
| Oracle Cloud | Always Free ARM VM is 4 cores / 24 GB, free permanently and no cold starts, but you set up Docker, a reverse proxy and TLS yourself. |

## Running it locally

```bash
# 1. weights (≈460 MB, from the repo's v3 release)
cd backend
python -m venv .venv && .venv/Scripts/activate     # source .venv/bin/activate on Unix
pip install -r requirements.txt
python download_models.py

# 2. secrets
cp .env.example .env        # then fill in the two keys

# 3. API
uvicorn main:app --reload --port 8000

# 4. UI, in a second terminal — vite proxies /analyze* to :8000
cd frontend && npm install && npm run dev
```

Or build the production image:

```bash
docker build -t truthshield .
docker run --rm -p 7860:7860 \
  -e NVIDIA_API_KEY=... -e GNEWS_API_KEY=... \
  truthshield
# → http://localhost:7860
```

## Configuration reference

| Variable | Default | Meaning |
|---|---|---|
| `NVIDIA_API_KEY` | — | Enables Agent / Both modes. Missing → falls back to model-only. |
| `GNEWS_API_KEY` | — | Enables live web verification. Missing → the agent reasons without web evidence. |
| `PORT` | `7860` | Listen port. |
| `MODELS_DIR` | `backend/models` | Where the checkpoints live. |
| `PHASE1_DIR` | `$MODELS_DIR/phase1_roberta_fulltune/best` | RoBERTa checkpoint. |
| `PHASE2_HEAD` | `$MODELS_DIR/phase2_fusion_head/fusion_head.pt` | Fusion head. |
| `FRONTEND_DIST` | `backend/frontend_dist` | Built SPA. Absent → API-only mode. |
| `ALLOWED_ORIGINS` | `*` | Comma-separated CORS allowlist. |
