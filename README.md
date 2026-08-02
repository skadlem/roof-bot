# Roof Bot

[![CI](https://github.com/skadlem/roof-bot/actions/workflows/ci.yml/badge.svg)](https://github.com/skadlem/roof-bot/actions/workflows/ci.yml)

**A WhatsApp sales manager for a roofing company.** It picks up every incoming message in Russian, runs a full SPIN sales conversation as a Gemini agent (tool-calling), prices the job live from your price list, and hands every confirmed lead to you in Google Sheets and on your own WhatsApp.

## What it does

- **Sells by itself** — Gemini AI drives the conversation with a real sales script (diagnosis → pain discovery → mirror presentation → objection handling → close), tuned for roofing: object type, build stage, roof area and geometry, deadlines.
- **Prices on the spot** — an agent tool (`lookup_pricing`) reads your price list (`prices.json`, ₸/m²) and answers exact prices mid-conversation; the total (price × area) is named and quoted before the close.
- **Understands voice messages** — customers can send audio; the bot downloads it from WhatsApp and transcribes it.
- **Writes like a human** — short messenger-style replies, one question per message, no lists, no emoji spam, no fake urgency. Follows up on objections without pressure.
- **Captures the full order** — name, material, color, area, final price, delivery address — and finalizes it in one message with the `create_lead` tool.
- **Delivers every lead to you** — each lead is appended to a Google Sheet for your CRM / bookkeeping and duplicated to the owner's WhatsApp.
- **Follows up cold leads** — APScheduler sends one polite reminder after 4 hours of silence, then leaves the customer alone.
- **Safe by default** — verifies the Meta webhook signature on every request, deduplicates webhook deliveries, rate-limits per customer, and logs every conversation to a per-phone-number file.

## How a conversation goes

1. A customer writes (or sends a voice message) to your WhatsApp number.
2. The bot introduces itself once — "Вы обратились в компанию МеталлКровля, я виртуальный менеджер по кровле" — and starts diagnosing: what's the object, what stage is the build, what area and shape, what's the deadline.
3. It finds what already annoyed the customer with other contractors (price, terms, trust) and mirrors their own words back in the offer.
4. Objections like "дорого" or "подумаю" get handled with a formula — agree → clarify → counter → small step — never pressure.
5. On confirmation it calculates the total and saves the lead (`create_lead` tool) — the tool validates the name and phone and appends the row to the sheet.
6. The lead lands in Google Sheets and on the owner's phone (template `new_order_notification`); the customer gets a clean goodbye.

## How it's built

```
Customer ──▶ WhatsApp Cloud API ──▶ FastAPI webhook ──▶ Gemini agent (text + voice)
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
- **Gemini API** — the sales brain: an agent loop (up to 3 turns per message) with two tools, driven by a full system-prompt sales script
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
   - `OWNER_PHONE_NUMBER` — owner's phone number, new leads are duplicated to it
2. Place `google_credentials.json` (Google Sheets service account) in the repo root.
3. Put your prices in `prices.json` (₸ per m²) — the bot quotes them in conversation.

## Knowledge base (RAG)

Two pipelines feed the bot. Write real facts in `kb/*.md` — one topic per file, one fact per paragraph (see `kb/README.md` for the format and what to fill in).

**`rag/` (new)** — chunks `kb/*.md` and your Google Sheets orders, embeds them with Gemini (`gemini-embedding-2`), and stores everything in a persistent ChromaDB store (`chroma_db/`, gitignored):

```bash
venv\Scripts\python.exe -m rag.ingest
```

Idempotent: re-running only re-embeds changed sources (per-source sha256). Sheet rows get `client_id = phone`; kb facts are shared across all clients. This is the pipeline the bot uses at runtime: every text message retrieves the closest chunks for the client's phone and wraps the first agent turn with them as context (only chunks above the similarity threshold — `RAG_SIMILARITY_THRESHOLD` in `rag/prompts.py`). Try the pieces manually:

```bash
venv\Scripts\python.exe -m rag.retrieve "how much does metal tile cost" <client_id>
venv\Scripts\python.exe -m rag.ask "how much does metal tile cost" <client_id>      # step-2 flow: context → Gemini answer (no tools)
venv\Scripts\python.exe -m rag.agent_cli <client_id>                                # agent loop: multi-turn REPL, prints every tool call
```

Needs `GEMINI_API_KEY` and `SHEET_KEY` in `.env`, plus `google_credentials.json`.

**`build_kb.py` (legacy, no longer loaded)** — the previous retrieval path (`kb_embeddings.json`, `gemini-embedding-001`). The bot stopped loading it when `rag/` was wired in; keep the file only for reference.

If the knowledge base is missing or no chunk is above the threshold, retrieval is disabled: the bot never invents prices, services, timelines, delivery zones, or guarantees — it says it will check with a manager. Voice messages never trigger retrieval.

## Agent loop (tools)

Every message runs through a small agent loop (`rag/agent.py`): the first Gemini turn is grounded in the retrieved KB context (or raw audio parts for voice), and every `function_call` the model makes is executed and fed back — up to `MAX_TURNS` (3) calls to Gemini before the final text goes to the customer. The whole sequence runs under a per-phone lock, so one customer's turns never interleave; a shared sliding-window rate limiter (`GEMINI_RPM`, default 14) paces all calls against the API quota.

Two tools:

- `lookup_pricing(service)` — exact price per m² from `prices.json`, the same live file embedded in the system prompt (the copy in `kb/faq.md` is stale). The model is told to quote prices only through this tool.
- `create_lead(name, service, message)` — appends the lead row to the Google Sheet. The phone is optional: if the model doesn't pass one, the customer's own WhatsApp number is used (validated 7–15 digits). The prompt is hardened so an order means the call: name + material + area in the message counts as an order — no asking for confirmation, no "the manager will call you" without the call. A validation error goes back to the model to relay — a bad lead never reaches the sheet and the conversation stays open. A saved lead closes the session.

## Run

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

Point the WhatsApp webhook at `https://<your-domain>/webhook`. Running locally — expose with a tunnel (`ngrok http 8000`) and use the ngrok URL in the Meta dashboard; update it there if the URL changes.

## Evaluations

A golden set of 24 question/answer pairs (`evals/golden_set.jsonl`) runs the real bot prompt through the agent loop (`rag/agent.py`) without WhatsApp or Sheets (leads are validated but never written), then a Gemini judge scores each answer twice — grounded (nothing beyond the retrieved context / tool results) and correct (semantically matches the expected answer). The judge prompt lives in `evals/judge_prompt.txt` and can be edited without touching code.

```bash
$env:GEMINI_RPM = "10"          # PowerShell: keep under the free-tier 15 rpm burst limit
venv\Scripts\python.exe -m evals.run_evals
```

Prints a pass-rate table (overall and per kb/tools group) and the worst failed cases. Exit code 1 if grounded or correct is below 70%, 2 if cases couldn't be evaluated. Current: **grounded 96%, correct 79% (24 cases), exit 0** — kb 100%/100%, tools 95%/74%.

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
