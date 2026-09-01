"""Deploy Truth Shield on Modal.

Serves exactly what the Docker image serves - the Svelte bundle plus /analyze,
/analyze_stream and /health - on one URL, but scales to zero between visitors so
a portfolio link costs almost nothing.

    pip install modal
    modal setup                                   # one-time browser auth
    cd frontend && npm ci && npm run build && cd ..
    modal secret create truth-shield-keys NVIDIA_API_KEY=... GNEWS_API_KEY=...
    modal deploy modal_app.py

The weights come from the repo's v3 GitHub release and are baked into the image
at build time, so a cold container never waits on a 460 MB download.
"""

from pathlib import Path

import modal

APP_NAME = "truth-shield"
REMOTE = "/app"

HERE = Path(__file__).parent
FRONTEND_DIST = HERE / "frontend" / "dist"

# Only meaningful at deploy time. Modal also imports this module inside the
# container, where __file__ is /root and the local tree does not exist - the
# bundle is already baked into the image by then.
if modal.is_local() and not FRONTEND_DIST.is_dir():
    raise SystemExit(
        f"{FRONTEND_DIST} is missing - build the UI before deploying:\n"
        "    cd frontend && npm ci && npm run build"
    )

image = (
    modal.Image.debian_slim(python_version="3.11")
    # CPU-only torch. The default PyPI wheel drags in ~2.5 GB of CUDA libraries
    # that a CPU container can never use.
    .run_commands(
        "pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu"
    )
    .add_local_file(HERE / "backend" / "requirements.txt", "/tmp/requirements.txt", copy=True)
    .run_commands("pip install --no-cache-dir -r /tmp/requirements.txt")
    # Backend source, named file by file - backend/ also holds a local .venv and
    # the fetched weights, and neither belongs in the image.
    .add_local_file(HERE / "backend" / "main.py", f"{REMOTE}/main.py", copy=True)
    .add_local_file(HERE / "backend" / "download_models.py", f"{REMOTE}/download_models.py", copy=True)
    .add_local_dir(HERE / "backend" / "static", f"{REMOTE}/static", copy=True)
    .add_local_dir(HERE / "backend" / "templates", f"{REMOTE}/templates", copy=True)
    .add_local_dir(FRONTEND_DIST, f"{REMOTE}/frontend_dist", copy=True)
    # Everything a first request would otherwise have to fetch.
    .run_commands(
        f"cd {REMOTE} && python download_models.py",
        "python -c \"import nltk; nltk.download('punkt'); nltk.download('punkt_tab')\"",
        "python -c \"from transformers import GPT2TokenizerFast, GPT2LMHeadModel; "
        "GPT2TokenizerFast.from_pretrained('gpt2'); GPT2LMHeadModel.from_pretrained('gpt2')\"",
    )
    .env(
        {
            "MODELS_DIR": f"{REMOTE}/models",
            "FRONTEND_DIST": f"{REMOTE}/frontend_dist",
            "TOKENIZERS_PARALLELISM": "false",
            # Modal bills max(request, actual) and lets containers burst far above
            # the request. Torch otherwise sizes its pool to every visible core,
            # which would be billed. Pin it to the hard limit set below.
            "OMP_NUM_THREADS": "2",
            "MKL_NUM_THREADS": "2",
        }
    )
)

app = modal.App(APP_NAME)


@app.cls(
    image=image,
    # (request, hard limit). Idle containers bill at the request, so keeping it
    # at 1 makes waiting cheap, while the limit still lets a live request use
    # two cores. Without the limit a burst would be billed at up to 16.
    cpu=(1, 2),
    memory=2048,          # measured peak is ~1.6 GB on a 512-token phase-2 request
    scaledown_window=60,  # idle time is the dominant cost on a low-traffic demo
    timeout=600,
    enable_memory_snapshot=True,
    secrets=[modal.Secret.from_name("truth-shield-keys")],
)
@modal.concurrent(max_inputs=4)
class TruthShield:
    @modal.enter(snap=True)
    def load(self):
        """Import the app once, then let Modal snapshot the loaded weights.

        Restoring that snapshot is what keeps cold starts to a few seconds
        instead of the ~40 s it takes to read 411 MB of weights off disk.
        """
        import sys

        sys.path.insert(0, REMOTE)
        import main

        self.api = main.app

    @modal.asgi_app()
    def web(self):
        return self.api
