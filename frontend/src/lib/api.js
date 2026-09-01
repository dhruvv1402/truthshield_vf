// Where the FastAPI backend lives.
//
// Default is same-origin: the production container serves this bundle from the
// same FastAPI app, and `npm run dev` proxies /analyze* to :8000 (vite.config.js).
// Set VITE_API_URL at build time only when the UI is hosted apart from the API
// (e.g. a Vercel frontend pointing at a Hugging Face Space).
export const API_BASE = (import.meta.env.VITE_API_URL ?? '').replace(/\/+$/, '');

export const apiUrl = (path) => `${API_BASE}${path}`;
