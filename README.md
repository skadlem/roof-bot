# Roof Bot

[![CI](https://github.com/skadlem/roof-bot/actions/workflows/ci.yml/badge.svg)](https://github.com/skadlem/roof-bot/actions/workflows/ci.yml)

WhatsApp bot for roofing installation leads. Talks to customers in Russian, collects lead details (roofing type, area, phone number), saves leads and sends follow-ups to warm leads.

## How it works

- **WhatsApp Cloud API** (Meta) — receiving/sending messages via webhook
- **Gemini API** — AI conversation with the customer
- **Google Sheets (gspread)** — lead export
- **APScheduler** — follow-ups for unanswered leads

## Installation

```bash
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

## Setup

1. Copy `.env.example` to `.env` and fill in the values:
   - `GEMINI_API_KEY` — Google Gemini key
   - `WA_TOKEN`, `WA_PHONE_ID`, `WA_VERIFY_TOKEN`, `WA_APP_SECRET` — WhatsApp Cloud API settings
   - `OWNER_PHONE_NUMBER` — owner's phone number, leads are duplicated to it
2. Place `google_credentials.json` (Google Sheets service account) in the repo root.

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

## Note

Data files contain customer personal data (phone numbers, conversations) — do not publish them.
