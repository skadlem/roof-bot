# Knowledge base (RAG)

The bot answers factual questions (services, timelines, delivery areas, guarantees) from this folder instead of guessing. Each fact is stored as a text chunk; at startup the bot finds the chunks closest to the customer's question and sends them to Gemini as context.

## Format

- One topic per file, Markdown (`kb/*.md`), e.g. `kb/faq.md`.
- One fact = one paragraph. Keep paragraphs short (~300–500 characters) — a paragraph is the unit of retrieval.
- Markdown headers are just file organization — they are not indexed.
- Prices are **not** needed here: they already come from `prices.json` via the system prompt. The example section in `kb/faq.md` shows how a filled-in section looks.

## What to fill in

Real facts only. The current `kb/faq.md` is filled with **fake demo data** (services, timelines, delivery, guarantees, materials) so the bot can be tested on real phone numbers. Replace it with real company facts before production — anything you write here is what the bot promises to customers:

- **Services** — монтаж кровли, замена, ремонт, под ключ, что входит в работу.
- **Mounting timelines** — сколько дней занимает типовой объект, от чего зависит срок.
- **Delivery areas** — куда доставляете материалы, бесплатная ли доставка, самовывоз.
- **Guarantees** — гарантия на материалы и работы, что именно покрывает.
- **FAQ** — типовые вопросы клиентов и честные ответы.
- **Material details** — толщина металла, покрытие, отличия металлочерепицы / профнастила / фальца / мягкой кровли.

Anything you write here is what the bot promises to customers — if you don't fill it in, the bot honestly says it will check with a manager instead of inventing facts.

## Rebuild after editing

```bash
venv\Scripts\python.exe -m rag.ingest
```

Requires `GEMINI_API_KEY` (or `GOOGLE_API_KEY`) in `.env`. The ingest chunks every `kb/*.md`, embeds the chunks, and writes them into `chroma_db/` (gitignored). It is idempotent: re-running only re-embeds changed files. The bot picks the change up immediately.

If the knowledge base is empty or the file is missing, retrieval is disabled and the bot falls back to the safe behavior: it never invents prices, services, timelines, delivery zones, or guarantees.
