# Roof Bot

[![CI](https://github.com/skadlem/roof-bot/actions/workflows/ci.yml/badge.svg)](https://github.com/skadlem/roof-bot/actions/workflows/ci.yml)

**A WhatsApp sales manager for a roofing company.** It picks up every incoming message in Russian, runs a full SPIN sales conversation, prices the job live from your price list, and hands every confirmed order to you in Google Sheets and on your own WhatsApp.

## What it does

- **Sells by itself** — Gemini AI drives the conversation with a real sales script (diagnosis → pain discovery → mirror presentation → objection handling → close), tuned for roofing: object type, build stage, roof area and geometry, deadlines.
- **Prices on the spot** — loads your price list (`prices.json`, ₸/m²) and calculates the customer's total mid-conversation (price × area), named and quoted before the close.
- **Understands voice messages** — customers can send audio; the bot downloads it from WhatsApp and transcribes it.
- **Writes like a human** — short messenger-style replies, one question per message, no lists, no emoji spam, no fake urgency. Follows up on objections without pressure.
- **Captures the full order** — name, material, color, area, final price, delivery address — and finalizes it in one message.
- **Delivers every lead to you** — each order is duplicated to the owner's WhatsApp and appended to a Google Sheet for your CRM / bookkeeping.
- **Follows up cold leads** — APScheduler sends one polite reminder after 4 hours of silence, then leaves the customer alone.
- **Safe by default** — verifies the Meta webhook signature on every request, deduplicates webhook deliveries, rate-limits per customer, and logs every conversation to a per-phone-number file.

## How a conversation goes

1. A customer writes (or sends a voice message) to your WhatsApp number.
2. The bot introduces itself once — "Вы обратились в компанию МеталлКровля, я виртуальный менеджер по кровле" — and starts diagnosing: what's the object, what stage is the build, what area and shape, what's the deadline.
3. It finds what already annoyed the customer with other contractors (price, terms, trust) and mirrors their own words back in the offer.
4. Objections like "дорого" or "подумаю" get handled with a formula — agree → clarify → counter → small step — never pressure.
5. On confirmation it calculates the total and creates the order.
6. The order lands in Google Sheets and on the owner's phone; the customer gets a clean goodbye.

## How it's built

```
Customer ──▶ WhatsApp Cloud API ──▶ FastAPI webhook ──▶ Gemini (text + voice)
    ▲                                   │
    └──────────── reply via API ◄───────┘
                                        │
                    ┌───────────────────┴────────────────────┐
                    ▼                                        ▼
         Google Sheets (lead export)          Owner's WhatsApp (order copy)
                    ▲
                    └── APScheduler: one follow-up after 4h of silence
```

- **FastAPI** — webhook endpoint, signature verification, message dedup
- **WhatsApp Cloud API (Meta)** — send/receive messages, download voice audio
- **Gemini API** — the sales brain, driven by a full system-prompt sales script
- **Google Sheets (gspread)** — lead export
- **APScheduler** — 4-hour follow-up for unanswered leads

## Installation

```bash
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

## Setup

1. Copy `.env.example` to `.env` and fill in the values:
   - `GEMINI_API_KEY` — Google Gemini key
   - `SHEET_KEY` — ID of the Google Sheet used for order export and RAG ingestion
   - `WA_TOKEN`, `WA_PHONE_ID`, `WA_VERIFY_TOKEN`, `WA_APP_SECRET` — WhatsApp Cloud API settings
   - `OWNER_PHONE_NUMBER` — owner's phone number, leads are duplicated to it
2. Place `google_credentials.json` (Google Sheets service account) in the repo root.
3. Put your prices in `prices.json` (₸ per m²) — the bot quotes them in conversation.

## Knowledge base (RAG)

Two pipelines feed the bot. Write real facts in `kb/*.md` — one topic per file, one fact per paragraph (see `kb/README.md` for the format and what to fill in).

**`rag/` (new)** — chunks `kb/*.md` and your Google Sheets orders, embeds them with Gemini (`gemini-embedding-2`), and stores everything in a persistent ChromaDB store (`chroma_db/`, gitignored):

```bash
venv\Scripts\python.exe -m rag.ingest
```

Idempotent: re-running only re-embeds changed sources (per-source sha256). Sheet rows get `client_id = phone`; kb facts are shared across all clients. This is the pipeline the bot uses at runtime: every text message retrieves the closest chunks for the client's phone and sends them to Gemini as context (only chunks above the similarity threshold — `RAG_SIMILARITY_THRESHOLD` in `rag/prompts.py`). Try it manually:

```bash
venv\Scripts\python.exe -m rag.retrieve "how much does metal tile cost" <client_id>
venv\Scripts\python.exe -m rag.ask "how much does metal tile cost" <client_id>   # bot flow: context → Gemini answer + sources
```

Needs `GEMINI_API_KEY` and `SHEET_KEY` in `.env`, plus `google_credentials.json`.

**`build_kb.py` (legacy, no longer loaded)** — the previous retrieval path (`kb_embeddings.json`, `gemini-embedding-001`). The bot stopped loading it when `rag/` was wired in; keep the file only for reference.

If the knowledge base is missing or no chunk is above the threshold, retrieval is disabled: the bot never invents prices, services, timelines, delivery zones, or guarantees — it says it will check with a manager. Voice messages never trigger retrieval.

## Run

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

Point the WhatsApp webhook at `https://<your-domain>/webhook`. Running locally — expose with a tunnel (`ngrok http 8000`) and use the ngrok URL in the Meta dashboard; update it there if the URL changes.

## Testing

```bash
pip install -r requirements-dev.txt
pytest
```

All tests are offline — external services (Gemini, WhatsApp, Google Sheets) are mocked. CI runs them on every push and pull request.

## Data files

Created at runtime and **not tracked by git** (in `.gitignore`):

| File | Contents |
|---|---|
| `open_leads.json` | Open leads with chat history |
| `seen_messages.json` | Processed message IDs |
| `chats_logs/` | Per-phone-number chat logs |
| `chroma_db/` | RAG vector store (rebuild with `python -m rag.ingest`) |
| `kb_embeddings.json` | Legacy RAG store (rebuild with `build_kb.py`) |

## Note

Data files contain customer personal data (phone numbers, conversations) — do not publish them.
