# Truth Shield 🛡️

**Truth Shield** is an advanced **Agentic Threat Intelligence & Fake News Detection platform**, initially developed under the codename *VIGIL-AI*. It features a multi-layered verification system that seamlessly balances fine-tuned Machine Learning models with live web verification provided by AI Agents.

Our ecosystem is designed for structural linguistic analysis of text (fake news detection) through state-of-the-art machine learning algorithms and reasoning engines.

---

## 🔥 Key Features & Capabilities

### 1. Fake News & Threat Intelligence Analysis
- **RoBERTa Linguistic Backbone**: A custom-trained Natural Language Processing pipeline that captures nuances in fake news and malicious text.
- **Agentic Live Web Verification**: A dedicated AI agent actively queries live sources to verify facts, balancing synthetic reasoning (40%) with live web verification (60%).
- **Granular Controls**: Granular chat analysis interface allowing users to dynamically toggle between the custom-trained static model, the live web-aware AI agent, or a combination of both.

---

## 🏗️ Technology Stack

| Component         | Technology |
|-------------------|------------|
| **Frontend**      | Svelte, Vite, Vercel/Node |
| **Backend**       | Python, FastAPI, Uvicorn, WebSockets |
| **Machine Learning** | PyTorch, HuggingFace (RoBERTa), NLTK |
| **LLMs / Agents** | Ollama, Nvidia Gemma API |
| **Data Processing**| Pandas, Scikit-Learn |

---

## 📂 Project Structure

```bash
AIML-PROJECT-CSET312/
│
├── Dockerfile                # Single-image build: Svelte bundle + FastAPI + weights
├── DEPLOYMENT.md             # Hosting guide (Hugging Face Spaces, Vercel, others)
│
├── backend/                  # FastAPI backend server
│   ├── main.py               # /analyze, /analyze_stream, /health
│   ├── download_models.py    # Fetches the trained weights from the v3 release
│   └── requirements.txt      # Python dependencies
│
├── frontend/                 # Svelte-based Web Application
│   ├── src/                  # Svelte components & logic
│   ├── public/               # Static assets (including Truth Shield logo)
│   └── package.json          # Node.js dependencies
│
├── Datasets/                 # Local data sources (Kaggle, FakeNewsCorpus)
├── Notebooks/                # Jupyter Notebooks for data gathering & preprocessing
├── Model_desgin_vX/          # Historical architecture design prototypes
└── project_structure.txt     # Complete directory manifest
```

---

## ⚙️ Detailed Installation & Setup

### 1. Prerequisites
Ensure you have the following installed on your machine:
- **Python 3.10+**
- **Node.js 18+** & NPM
- **Git**

### 2. Backend Environment (FastAPI)

We recommend using `uv` or `venv` to isolate the backend environment. 

```bash
# Clone the repository
git clone <your-repo-url>
cd AIML-PROJECT-CSET312

# Navigate to the backend directory
cd backend

# Create and activate a Virtual Environment
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Mac/Linux:
source .venv/bin/activate

# Install strictly required dependencies
pip install -r requirements.txt

# Fetch the trained weights (~460 MB, from the v3 release)
python download_models.py
```

> The phase-1 RoBERTa checkpoint and phase-2 fusion head are published as
> [release assets](https://github.com/Gyaanendra/AIML-PROJECT-CSET312/releases/tag/v3),
> not committed to the tree. `download_models.py` pulls and unpacks both into
> `backend/models/`.

#### Environment Variables Config (`.env`)
You must configure your `.env` file in the `/backend` directory before running. Do not commit credentials to Git.
```env
NVIDIA_API_KEY=your_nvidia_build_key   # reasoning agent  -> build.nvidia.com
GNEWS_API_KEY=your_gnews_key           # live verification -> gnews.io
```

Both are optional: without them the API still serves the trained classifier and
says so in the UI. See `backend/.env.example` for the full list.

#### Run the Backend Server
```bash
uvicorn main:app --reload --port 8000
```
*The FastAPI backend will now be actively listening on `http://localhost:8000`. WebSocket streams connect on `ws://localhost:8000`.*

### 3. Frontend Environment (Svelte)

The frontend is the analysis interface: the detector view, streamed verdicts, and per-token attention highlighting.

```bash
# Open a new terminal and navigate to the frontend
cd AIML-PROJECT-CSET312/frontend

# Install Node dependencies
npm install

# Start the Vite development server
npm run dev
```
*The dashboard will be accessible via your browser at the URL provided by Vite (usually `http://localhost:5173` or similar).*

---

## 📓 Model Training & Data Preparation

If you intend to adjust the base models, work happens within the `Notebooks/` directory.

- **`data_gathering.ipynb`**: Responsible for API connections and raw data ingestion.
- **`Preprocessing.ipynb`**: Responsible for NLTK tokenization, removing stop-words, scaling, and preparing unified CSVS for custom training pipelines.

---

## 🚀 Deployment

The whole app ships as one Docker image — the build compiles the Svelte bundle
and FastAPI serves it alongside the API, so a single URL is the entire product.

```bash
docker build -t truthshield .
docker run --rm -p 7860:7860 -e NVIDIA_API_KEY=... -e GNEWS_API_KEY=... truthshield
```

See **[DEPLOYMENT.md](DEPLOYMENT.md)** for hosted options, sizing (~2 GB RAM),
and the full configuration reference.

---

## 📜 Legal & License

This project is licensed under the **MIT License** (see `LICENSE` file for details). Please note that datasets obtained from Kaggle or external sources are subject to their respective proprietary licensing.
