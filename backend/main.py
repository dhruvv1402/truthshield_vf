import re
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import json
import asyncio
import os
import requests
import urllib.parse
from dotenv import load_dotenv

load_dotenv()

from langchain_nvidia_ai_endpoints import ChatNVIDIA
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    GPT2TokenizerFast,
    GPT2LMHeadModel,
)
import nltk

try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt", quiet=True)

try:
    nltk.data.find("tokenizers/punkt_tab")
except LookupError:
    nltk.download("punkt_tab", quiet=True)

# ── Paths ──────────────────────────────────────────────────────────────────────
# Everything resolves off this file, not the working directory, so the server
# behaves the same started from ./backend, from / in a container, or under a
# process manager.
BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = Path(os.getenv("MODELS_DIR", BASE_DIR / "models"))

PHASE1_DIR = os.getenv("PHASE1_DIR", str(MODELS_DIR / "phase1_roberta_fulltune" / "best"))
PHASE2_HEAD = os.getenv("PHASE2_HEAD", str(MODELS_DIR / "phase2_fusion_head" / "fusion_head.pt"))

# Built Svelte bundle. When it is present the API serves the UI too, so a single
# URL is the whole product.
FRONTEND_DIST = Path(os.getenv("FRONTEND_DIST", BASE_DIR / "frontend_dist"))

# NVIDIA retires hosted models on a schedule - gemma-3n-e2b-it went end-of-life
# on 2026-07-27 and now answers 410 - so keep this swappable without a redeploy.
AGENT_MODEL = os.getenv("AGENT_MODEL", "openai/gpt-oss-20b")

if not Path(PHASE1_DIR).exists():
    raise RuntimeError(
        f"Phase-1 weights not found at {PHASE1_DIR}. "
        "Fetch them first:  cd backend && python download_models.py"
    )

app = FastAPI(title="Truth Shield — Fake News Detector")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in os.getenv("ALLOWED_ORIGINS", "*").split(",") if o.strip()],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

# ── Models ─────────────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained(PHASE1_DIR, use_fast=False)
roberta = AutoModelForSequenceClassification.from_pretrained(
    PHASE1_DIR, num_labels=2, output_attentions=True
).to(device)
roberta.eval()


class FusionHead(nn.Module):
    """Mirrors the phase-2 training definition.

    See Model_desgin_v3/Model/Scripts/phase2_rtxa6000.py - the shapes here have
    to match the released fusion_head.pt exactly or load_state_dict fails.
    """

    def __init__(self, semantic_dim=768, stat_dim=3, hidden=256, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(semantic_dim + stat_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 2),
        )

    def forward(self, semantic, stat):
        return self.net(torch.cat([semantic, stat], dim=-1))


fusion_head = None
# Training-set mean/std for the three hand-built statistical features, saved
# next to the weights. Inference must reuse them: normalising a single sample
# against itself collapses every feature to exactly zero.
stat_mean = np.zeros(3, dtype=np.float32)
stat_std = np.ones(3, dtype=np.float32)

if Path(PHASE2_HEAD).exists():
    ckpt = torch.load(PHASE2_HEAD, map_location=device, weights_only=False)
    fusion_head = FusionHead(
        semantic_dim=ckpt.get("semantic_dim", 768),
        stat_dim=ckpt.get("stat_dim", 3),
        hidden=ckpt.get("hidden", 256),
    ).to(device)
    fusion_head.load_state_dict(ckpt["head_state"])
    fusion_head.eval()
    stat_mean = np.asarray(ckpt["stat_mean"], dtype=np.float32)
    stat_std = np.asarray(ckpt["stat_std"], dtype=np.float32)

gpt2_tok = GPT2TokenizerFast.from_pretrained("gpt2")
gpt2_tok.pad_token = gpt2_tok.eos_token
gpt2_mod = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
gpt2_mod.eval()

# ── Helpers ────────────────────────────────────────────────────────────────────

def clean_word(word: str) -> str:
    word = word.replace("\u00e2\u0080\u0099", "'")
    word = word.replace("\u00e2\u0080\u009c", '"')
    word = word.replace("\u00e2\u0080\u009d", '"')
    word = word.replace("\u010c\u201c", '"')
    word = word.replace("\u010c\u2122", "'")
    word = word.replace("\u00e2\u0122\u017e", '"')
    word = word.replace("\u00e2\u0122\u00be", '"')
    word = word.replace("\u00e2\u0122\u0080", "...")
    word = word.replace("\u00e2\u0122\u0081", "-")
    word = word.replace("\u00c4\u0141", "g")
    word = word.replace("\u00c4\u00b1", "i")
    word = re.sub(r"[^\x00-\x7F]+", "", word)
    word = re.sub(r" +", " ", word).strip()
    return word


def score_to_tier(score: float) -> int:
    if score > 0.92:
        return 4
    elif score > 0.78:
        return 3
    elif score > 0.62:
        return 2
    elif score > 0.48:
        return 1
    return 0


def build_token_highlights(attentions, enc):
    input_ids = enc["input_ids"][0].cpu().tolist()
    tokens = tokenizer.convert_ids_to_tokens(input_ids)
    real_len = sum(enc["attention_mask"][0].cpu().tolist())

    seq_len = attentions[0][0].shape[-1]
    rollout = torch.eye(seq_len).to(attentions[0][0].device)
    for layer_attn in attentions:
        avg_heads = layer_attn[0].mean(dim=0)
        aug = avg_heads + torch.eye(seq_len).to(avg_heads.device)
        aug = aug / aug.sum(dim=-1, keepdim=True)
        rollout = torch.matmul(aug, rollout)

    token_importance = rollout[0].cpu().numpy()
    tokens = tokens[1:real_len - 1]
    scores = token_importance[1:real_len - 1]

    ranks = np.argsort(np.argsort(scores))
    scores = ranks / (len(ranks) - 1 + 1e-8)

    result = []
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        score = float(scores[i])
        word = tok.replace("\u0120", "").replace("\u2581", "")

        while (
            i + 1 < len(tokens)
            and not tokens[i + 1].startswith("\u0120")
            and not tokens[i + 1].startswith("\u2581")
        ):
            i += 1
            word += tokens[i].replace("\u0120", "").replace("\u2581", "")
            score = max(score, float(scores[i]))

        word = clean_word(word)
        if word:
            result.append({
                "word": word,
                "tier": score_to_tier(score),
                "leading_space": tok.startswith("\u0120") or tok.startswith("\u2581"),
            })
        i += 1

    return result


@torch.no_grad()
def compute_stat_features(text: str):
    from nltk.tokenize import sent_tokenize
    sents = sent_tokenize(text)
    sent_var = float(np.var([len(s.split()) for s in sents])) if len(sents) > 1 else 0.0
    loss = 0.0
    if len(text.split()) >= 2:
        enc = gpt2_tok(
            text, return_tensors="pt", truncation=True, max_length=512, padding=True
        ).to(device)
        loss = gpt2_mod(**enc, labels=enc.input_ids).loss.item()
    return np.array([[loss, sent_var, -loss]], dtype=np.float32)


def normalize_stat(stat_raw):
    """Apply the normalisation the fusion head was fitted with."""
    return (stat_raw - stat_mean) / stat_std


# ── Routes ─────────────────────────────────────────────────────────────────────

class AnalyzeRequest(BaseModel):
    text: str
    model: str = "phase2"
    # 'model' = ML only | 'agent' = LLM agent only | 'both' = full pipeline
    agent_mode: str = "both"


@app.post("/analyze")
@torch.no_grad()
def analyze(req: AnalyzeRequest):
    text = req.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Text cannot be empty.")

    enc = tokenizer(
        text, truncation=True, max_length=512, padding=True, return_tensors="pt"
    ).to(device)

    if req.model == "phase2" and fusion_head is not None:
        roberta_out = roberta.roberta(**enc, output_attentions=True)
        cls_emb = roberta_out.last_hidden_state[:, 0, :].cpu()
        attentions = roberta_out.attentions
        stat_norm = normalize_stat(compute_stat_features(text))
        logits = fusion_head(
            cls_emb.to(device),
            torch.tensor(stat_norm, dtype=torch.float32).to(device),
        )
        probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
    else:
        out = roberta(**enc)
        probs = torch.softmax(out.logits, dim=-1).cpu().numpy()[0]
        attentions = out.attentions

    label = "FAKE" if probs.argmax() == 1 else "REAL"
    tokens = build_token_highlights(attentions, enc)

    return {
        "label": label,
        "prob_real": round(float(probs[0]), 4),
        "prob_fake": round(float(probs[1]), 4),
        "tokens": tokens,
    }


@app.post("/analyze_stream")
async def analyze_stream(req: AnalyzeRequest):
    text = req.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Text cannot be empty.")

    def run_local_ml():
        with torch.no_grad():
            enc = tokenizer(
                text, truncation=True, max_length=512, padding=True, return_tensors="pt"
            ).to(device)

            if req.model == "phase2" and fusion_head is not None:
                roberta_out = roberta.roberta(**enc, output_attentions=True)
                cls_emb = roberta_out.last_hidden_state[:, 0, :].cpu()
                attentions = roberta_out.attentions
                stat_norm = normalize_stat(compute_stat_features(text))
                logits = fusion_head(
                    cls_emb.to(device),
                    torch.tensor(stat_norm, dtype=torch.float32).to(device),
                )
                probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
            else:
                out = roberta(**enc)
                probs = torch.softmax(out.logits, dim=-1).cpu().numpy()[0]
                attentions = out.attentions

            label = "FAKE" if probs.argmax() == 1 else "REAL"
            tokens = build_token_highlights(attentions, enc)
            return label, probs, tokens

    def search_web_claims(claims):
        try:
            gnews_key = os.getenv("GNEWS_API_KEY")
            if not gnews_key:
                return "Search failed: GNEWS_API_KEY is not configured."
            query = urllib.parse.quote(claims.strip())
            url = f"https://gnews.io/api/v4/search?q={query}&lang=en&max=3&apikey={gnews_key}"
            res = requests.get(url, timeout=10)
            if res.status_code == 200:
                data = res.json()
                results = ""
                for article in data.get("articles", []):
                    results += f"Title: {article['title']}\nSnippet: {article['description']}\n\n"
                return results if results else "No strict verifications found."
            return f"Search API returned {res.status_code}"
        except Exception as e:
            return f"Search failed: {str(e)}"

    agent_mode = req.agent_mode  # 'model' | 'agent' | 'both'

    # A deployment without an LLM key still has a working classifier, so fall
    # back to local scoring instead of failing the whole request.
    llm_unavailable = agent_mode != "model" and not os.getenv("NVIDIA_API_KEY")
    if llm_unavailable:
        agent_mode = "model"

    async def event_generator():
        # ── MODE: model only ────────────────────────────────────────────────────
        if agent_mode == "model":
            if llm_unavailable:
                yield json.dumps({
                    "type": "status",
                    "message": "Live agent offline (NVIDIA_API_KEY not set) - scoring with the local model only.",
                }) + "\n\n"
            yield json.dumps({"type": "status", "message": "Cross-referencing with Custom RoBERTa Backbone..."}) + "\n\n"
            label, probs, tokens_result = await asyncio.to_thread(run_local_ml)
            verdict_data = {
                "type": "verdict",
                "label": label,
                "prob_real": round(float(probs[0]), 4),
                "prob_fake": round(float(probs[1]), 4),
            }
            yield json.dumps(verdict_data) + "\n\n"
            for tok in tokens_result:
                await asyncio.sleep(0.01)
                yield json.dumps({"type": "token", "token": tok}) + "\n\n"
            return

        # ── Shared LLM client (used by 'agent' and 'both') ──────────────────────
        client = ChatNVIDIA(
            model=AGENT_MODEL,
            api_key=os.getenv("NVIDIA_API_KEY"),
            temperature=0.8,
            top_p=0.9,
            max_completion_tokens=2048,
        )

        # The upstream LLM is a third party with variable latency, so calls to it
        # are contained: a failure degrades to local scoring rather than tearing
        # down a response the client has already begun reading.
        agent_ok = True
        claims, web_context = "", ""
        try:
            yield json.dumps({"type": "status", "message": "Extracting search parameters..."}) + "\n\n"
            extract_prompt = f"Extract 2 to 4 crucial keywords identifying the core event from this text to use as a search query. Output ONLY the keywords separated by spaces. DO NOT use bullet points or extra text:\n\n{text[:1500]}"
            claims_res = await asyncio.to_thread(client.invoke, [{"role": "user", "content": extract_prompt}])
            claims = claims_res.content

            yield json.dumps({"type": "status", "message": "Verifying facts against live web data..."}) + "\n\n"
            web_context = await asyncio.to_thread(search_web_claims, claims)
        except Exception as exc:
            agent_ok = False
            yield json.dumps({
                "type": "status",
                "message": f"Live agent unavailable ({type(exc).__name__}) - scoring with the local model instead.",
            }) + "\n\n"

        # ── MODE: both — also run ML scoring ───────────────────────────────────
        label, probs, tokens_result = None, None, []
        if agent_mode == "both" or not agent_ok:
            yield json.dumps({"type": "status", "message": "Cross-referencing with Custom RoBERTa Backbone..."}) + "\n\n"
            label, probs, tokens_result = await asyncio.to_thread(run_local_ml)

        # Emit verdict + token highlights whenever the classifier ran, before any
        # early return, so a degraded request still delivers a usable result.
        if probs is not None:
            verdict_data = {
                "type": "verdict",
                "label": label,
                "prob_real": round(float(probs[0]), 4),
                "prob_fake": round(float(probs[1]), 4),
            }
            yield json.dumps(verdict_data) + "\n\n"
            for tok in tokens_result:
                await asyncio.sleep(0.01)
                yield json.dumps({"type": "token", "token": tok}) + "\n\n"

        if not agent_ok:
            return

        yield json.dumps({"type": "status", "message": "Synthesizing Threat Intelligence..."}) + "\n\n"

        # Build the LLM final prompt
        web_has_hits = (
            "No strict verifications found" not in web_context
            and "Search failed" not in web_context
        )

        if agent_mode == "both" and probs is not None:
            ml_conf = max(probs[0], probs[1])
            ml_section = f"""## ML STRUCTURAL ANALYSIS (40% weight)
- Prediction: {label} | Confidence: {ml_conf:.2%}
- Note: Trained on pre-2024 data — moderate caution for highly recent breaking events.

## REASONING RULES
1. Web evidence CONFIRMS claims → lean REAL, use ML as a confidence booster.
2. Web evidence CONTRADICTS claims → lean FAKE regardless of ML output.
3. Web evidence is ABSENT but ML confidence is HIGH (>85%) → lean toward ML verdict, note temporal lag.
4. Both signals agree → high-confidence verdict in that direction.
5. Signals conflict with low confidence → INCONCLUSIVE."""
            weight_note = "- Live Web Verification (GNews): 60% weight — PRIMARY signal.\n- Custom Trained ML Model (RoBERTa + Fusion Head): 40% weight — SECONDARY signal."
        else:
            ml_section = "## NOTE\nNo ML model output available — base your verdict solely on the live web evidence."
            weight_note = "- Live Web Verification (GNews): 100% weight — sole signal for this analysis."

        final_prompt = f"""You are VIGIL-AI, an advanced threat intelligence system determining if a news article is REAL or FAKE.

## SCORING WEIGHTS
{weight_note}

## TARGET TEXT
"{text[:900]}"

## LIVE WEB EVIDENCE
{"CORROBORATING ARTICLES FOUND:" if web_has_hits else "NO CORROBORATING EVIDENCE FOUND in live news sources."}
{web_context}

{ml_section}

Write 2-3 sentences of weighted analysis, then conclude:
**VERDICT: [REAL / FAKE / INCONCLUSIVE]** — one decisive bottom-line sentence.
"""

        try:
            async for chunk in client.astream([{"role": "user", "content": final_prompt}]):
                if hasattr(chunk, "additional_kwargs") and "reasoning_content" in chunk.additional_kwargs:
                    reasoning = chunk.additional_kwargs["reasoning_content"]
                    if reasoning:
                        yield json.dumps({"type": "llm_chunk", "reasoning": reasoning}) + "\n\n"
                if chunk.content:
                    yield json.dumps({"type": "llm_chunk", "content": chunk.content}) + "\n\n"
        except Exception as exc:
            yield json.dumps({
                "type": "status",
                "message": f"Agent synthesis interrupted ({type(exc).__name__}).",
            }) + "\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.get("/health")
def health():
    return {
        "status": "ok",
        "device": str(device),
        "phase2_head": fusion_head is not None,
        "agent": bool(os.getenv("NVIDIA_API_KEY")),
        "web_search": bool(os.getenv("GNEWS_API_KEY")),
        "agent_model": AGENT_MODEL,
    }


# -- Frontend --------------------------------------------------------------
# Registered last so every API route above still wins the match; anything else
# falls through to the SPA, which does its own client-side routing (/chat).
if FRONTEND_DIST.is_dir():
    _DIST = FRONTEND_DIST.resolve()

    @app.get("/{full_path:path}", include_in_schema=False)
    async def spa(full_path: str):
        candidate = (_DIST / full_path).resolve()
        if full_path and candidate.is_file() and _DIST in candidate.parents:
            return FileResponse(candidate)
        return FileResponse(_DIST / "index.html")

else:

    @app.get("/", include_in_schema=False)
    async def index():
        return {
            "service": "Truth Shield API",
            "status": "running",
            "endpoints": ["/analyze", "/analyze_stream", "/health", "/docs"],
        }
