---
title: Truth Shield
emoji: 🛡️
colorFrom: indigo
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
license: mit
short_description: Dual-phase RoBERTa + live-web agent for fake news detection
---

# Truth Shield 🛡️

Agentic threat-intelligence and fake-news detection. A fine-tuned RoBERTa
backbone scores the linguistic structure of a claim, an LLM agent checks it
against live news, and the two signals are weighted into a single verdict.

**The UI is served from this Space** — open it and paste an article into the
Detector.

| Mode | What runs |
|------|-----------|
| **Model** | Fine-tuned RoBERTa (phase 1) or the RoBERTa + statistical fusion head (phase 2) |
| **Agent** | LLM extracts the core claim, queries live news, reasons over the evidence |
| **Both** | Live web evidence at 60% weight, model score at 40% |

Token highlighting comes from attention rollout across all 12 layers, so you
can see which spans drove the verdict.

## Configuration

Set these as **Space secrets** (Settings → Variables and secrets):

| Secret | Purpose | Where to get it |
|--------|---------|-----------------|
| `NVIDIA_API_KEY` | Reasoning agent | [build.nvidia.com](https://build.nvidia.com) |
| `GNEWS_API_KEY` | Live web verification | [gnews.io](https://gnews.io) |

Without them the Space still runs — it falls back to local model scoring only.

## API

`POST /analyze` and `POST /analyze_stream` (newline-delimited JSON stream).
`GET /health` reports device and which capabilities are configured.

Source: <https://github.com/dhruvv1402/truthshield_vf>
