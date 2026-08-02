# Roof Bot

WhatsApp-бот для приёма заявок на монтаж кровли. Отвечает клиенту на русском, собирает данные о заявке (тип кровли, площадь, телефон), сохраняет лидов и отправляет follow-up по «тёплым» лидам.

## Как работает

- **WhatsApp Cloud API** (Meta) — приём/отправка сообщений через вебхук
- **Gemini API** — ИИ-диалог с клиентом
- **Google Sheets (gspread)** — запись заявок
- **APScheduler** — follow-up для лидов без ответа

## Установка

```bash
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

## Настройка

1. Скопируйте `.env` — все секреты берутся из него (см. `.env.example`):
   - `GEMINI_API_KEY` — ключ Google Gemini
   - `WA_TOKEN`, `WA_PHONE_ID`, `WA_VERIFY_TOKEN`, `WA_APP_SECRET` — настройки WhatsApp Cloud API
   - `OWNER_PHONE_NUMBER` — телефон владельца, куда дублируются заявки
2. Поместите `google_credentials.json` (сервисный аккаунт Google Sheets) в корень репозитория.

## Запуск

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

Вебхук WhatsApp указывается на `https://<ваш-домен>/webhook`.

## Файлы данных

Создаются при работе бота и **не попадают в git** (в `.gitignore`):

| Файл | Содержимое |
|---|---|
| `open_leads.json` | Открытые лиды с историей переписки |
| `seen_messages.json` | ID обработанных сообщений |
| `chats_logs/` | Журналы переписок по номерам телефонов |

## Важно

Файлы данных содержат персональные данные клиентов (номера телефонов, переписку) — не публикуйте их.
