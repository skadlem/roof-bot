import os
import json
import time
import hmac
import hashlib
import asyncio
import threading
from contextlib import contextmanager
import requests
import gspread
import google.generativeai as genai
from fastapi import FastAPI, Request, HTTPException, Query
from dotenv import load_dotenv
from datetime import datetime

from apscheduler.schedulers.background import BackgroundScheduler

from rag.agent import gemini_rate_limiter, get_model, load_prices, run_agent
from rag.prompts import build_bot_system_prompt

LEADS_FILE = "open_leads.json"

# Реентрантная блокировка — защищает open_leads И chat_sessions от гонок,
# когда несколько сообщений (разных клиентов) обрабатываются параллельно в разных потоках.
state_lock = threading.RLock()

# Отдельные локи НА КАЖДЫЙ номер телефона — гарантируют, что два сообщения
# от ОДНОГО И ТОГО ЖЕ клиента обрабатываются строго по очереди, а не параллельно
# (иначе можно словить гонку внутри одного Gemini ChatSession). Записи
# рефкаунтятся: когда лок перестают использовать, запись удаляется — иначе
# словарь рос бы бесконечно (по записи на каждый когда-либо писавший номер).
_phone_locks_guard = threading.Lock()
_phone_locks: dict[str, "_PhoneLockEntry"] = {}


class _PhoneLockEntry:
    def __init__(self):
        self.lock = threading.Lock()
        self.users = 0


@contextmanager
def phone_lock(phone_number):
    """Контекстный менеджер: `with phone_lock(phone):` — тот же взаимоисключающий
    лок на номер, что и прежний get_phone_lock, плюс удаление записи из
    _phone_locks, когда лок освободил последний держатель."""
    with _phone_locks_guard:
        entry = _phone_locks.get(phone_number)
        if entry is None:
            entry = _phone_locks[phone_number] = _PhoneLockEntry()
        entry.users += 1
    entry.lock.acquire()
    try:
        yield
    finally:
        entry.lock.release()
        with _phone_locks_guard:
            entry.users -= 1
            if entry.users == 0:
                _phone_locks.pop(phone_number, None)


def load_leads():
    if os.path.exists(LEADS_FILE):
        try:
            with open(LEADS_FILE, "r", encoding="utf-8") as f:
                content = f.read()
                if not content:  # Если файл пустой
                    return {}
                data = json.loads(content)

                # Бэкаповская поддержка старого формата: {"phone": timestamp}
                normalized = {}
                for phone, value in data.items():
                    if isinstance(value, (int, float)):
                        normalized[phone] = {
                            "last_seen": float(value),
                            "followup_sent": False,
                            "history": [],
                        }
                    elif isinstance(value, dict):
                        # ожидаемый новый формат
                        normalized[phone] = {
                            "last_seen": float(value.get("last_seen", time.time())),
                            "followup_sent": bool(value.get("followup_sent", False)),
                            # история переписки нужна, чтобы после рестарта сервера
                            # можно было восстановить Gemini ChatSession, а не терять лида
                            "history": value.get("history", []),
                        }
                return normalized
        except json.JSONDecodeError:
            # Если файл поврежден, возвращаем пустой словарь
            return {}
    return {}


def save_leads(leads):
    with state_lock:
        tmp_path = LEADS_FILE + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(leads, f, ensure_ascii=False)
        # атомарная замена файла — исключает повреждение файла при падении
        # процесса ровно в момент записи
        os.replace(tmp_path, LEADS_FILE)


# Словарь для хранения открытых лидов:
# {"phone_number": {"last_seen": ts, "followup_sent": bool, "history": [...]}}
open_leads = load_leads()


# --- ДЕДУПЛИКАЦИЯ ВЕБХУКОВ ---
# Meta может повторно доставить один и тот же вебхук (сетевые ретраи), и тогда одно
# и то же сообщение клиента обработается дважды: клиент получит два одинаковых
# ответа, а в таблицу может уйти задвоенный лид. Храним ID уже обработанных
# сообщений на диске (переживает рестарт сервера) и отбрасываем повторы.
SEEN_MESSAGES_FILE = "seen_messages.json"
SEEN_MESSAGE_TTL = 24 * 60 * 60  # сутки — с запасом перекрывает окно ретраев Meta

_seen_messages_lock = threading.Lock()


def load_seen_messages():
    if os.path.exists(SEEN_MESSAGES_FILE):
        try:
            with open(SEEN_MESSAGES_FILE, "r", encoding="utf-8") as f:
                content = f.read()
                if not content:
                    return {}
                data = json.loads(content)
                now = time.time()
                # сразу отбрасываем протухшие записи при загрузке
                return {mid: ts for mid, ts in data.items() if now - ts <= SEEN_MESSAGE_TTL}
        except json.JSONDecodeError:
            return {}
    return {}


def save_seen_messages(seen):
    tmp_path = SEEN_MESSAGES_FILE + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(seen, f)
    os.replace(tmp_path, SEEN_MESSAGES_FILE)


seen_messages = load_seen_messages()


def is_duplicate_message(message_id: str | None) -> bool:
    """
    True, если сообщение с таким id уже обрабатывалось — значит, это повторная
    доставка вебхука, и её нужно отбросить, не запуская Gemini/WhatsApp повторно.
    """
    if not message_id:
        # У сообщения нет id (не должно происходить в норме) — не блокируем обработку,
        # просто не можем защититься от дублей именно для этого сообщения.
        return False

    now = time.time()
    with _seen_messages_lock:
        expired = [mid for mid, ts in seen_messages.items() if now - ts > SEEN_MESSAGE_TTL]
        for mid in expired:
            del seen_messages[mid]

        if message_id in seen_messages:
            return True

        seen_messages[message_id] = now
        save_seen_messages(seen_messages)
        return False


# Загружаем переменные из файла .env
load_dotenv()


# Загружаем цены при старте (load_prices переехал в rag/agent.py — его же
# использует инструмент lookup_pricing, чтобы не держать две копии прайса)
PRICES = load_prices()
prices_text = "\n".join([f"- {k}: {v} тг за кв.м" for k, v in PRICES.items()])

# --- БАЗА ЗНАНИЙ (RAG) ---
# Коллекция ChromaDB собирается rag/ingest.py из kb/*.md и Google Sheets.
# Контекст на первый ход агента подмешивает rag.agent.run_agent (первый ход
# агент-цикла — это шаг 2: retrieve → build_rag_message, порог в rag/prompts.py).

app = FastAPI()

# --- ИНИЦИАЛИЗАЦИЯ ---
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

WA_TOKEN = os.getenv("WA_TOKEN")
WA_PHONE_ID = os.getenv("WA_PHONE_ID")
VERIFY_TOKEN = os.getenv("WA_VERIFY_TOKEN")

# Секрет приложения Meta (App Dashboard -> Settings -> Basic -> App Secret).
# ЭТО НЕ WA_TOKEN. Нужен для проверки подписи входящих вебхуков.
WA_APP_SECRET = os.getenv("WA_APP_SECRET")

# Телефон владельца: уведомление о новом лиде (шаблон new_order_notification)
OWNER_PHONE_NUMBER = os.getenv("OWNER_PHONE_NUMBER")

# Google Sheets — ленивая инициализация: подключение происходит при ПЕРВОМ
# обращении (add_to_google_sheets), а не при старте. Так бот не падает при
# запуске, если google_credentials.json недоступен/битый, а сетевой сбой при
# экспорте заказа ловится общим try/except ниже.
_sheet = None
_sheet_lock = threading.Lock()


def get_sheet():
    global _sheet
    with _sheet_lock:
        if _sheet is None:
            gc = gspread.service_account(filename="google_credentials.json")
            sheet_key = os.getenv("SHEET_KEY")
            if not sheet_key:
                # ключ таблицы — секрет (репозиторий публичный): только из .env,
                # никаких захардкоженных фолбэков
                raise RuntimeError(
                    "SHEET_KEY не задан в .env — укажите ID таблицы для экспорта лидов"
                )
            _sheet = gc.open_by_key(sheet_key).sheet1
        return _sheet


# Создаем папку для логов, если её нет
if not os.path.exists("chats_logs"):
    os.makedirs("chats_logs")


def log_chat_to_file(phone_number, sender, text):
    """Сохраняет каждое сообщение в текстовый файл"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = f"[{timestamp}] {sender}: {text}\n"

    filename = f"chats_logs/chat_{phone_number}.txt"
    with open(filename, "a", encoding="utf-8") as f:
        f.write(log_entry)


# Хранилище активных сессий чатов (в памяти процесса).
# ВАЖНО: при запуске с несколькими uvicorn-воркерами (--workers > 1) у каждого
# воркера будет своя копия этого словаря — сообщения одного клиента могут попасть
# в разные процессы и потерять контекст. Для one-worker деплоя (типично для такого
# бота) это безопасно. Для мульти-воркер/мульти-инстанс продакшена нужен вынос
# состояния во внешнее хранилище (Redis/Postgres) — это уже выходит за рамки
# точечного фикса и требует отдельного рефакторинга.
chat_sessions = {}


def add_to_google_sheets(order_data):
    # Добавили колонку "Цена"
    row = [
        order_data.get("name", ""),
        order_data.get("phone", ""),
        order_data.get("material", ""),
        order_data.get("color", ""),
        order_data.get("area", ""),
        order_data.get("price", ""),  # НОВАЯ КОЛОНКА
        order_data.get("address", "Самовывоз"),
    ]
    try:
        get_sheet().append_row(row)
        return True
    except Exception as e:
        print(f"[GOOGLE SHEETS ERROR] {e}")
        return False


def _save_lead_to_sheets(order_data):
    """Хук записи лида для агент-цикла (rag.agent.run_agent).

    Таблица и уведомление владельцу: create_lead вызывает хук только при
    успешной валидации, так что сюда попадают только настоящие лиды.
    Владелец уведомляется ТОЛЬКО после реальной записи в таблицу — иначе
    он получит лид, которого в таблице нет. Возвращает успех, чтобы агент
    не закрывал сессию и не говорил клиенту «заказ оформлен» при сбое.
    """
    ok = add_to_google_sheets(order_data)
    if ok:
        send_whatsapp_to_owner(order_data)
    return ok


def download_whatsapp_media(media_id: str):
    """Скачивает медиафайл (аудио) с серверов WhatsApp"""
    url = f"https://graph.facebook.com/v18.0/{media_id}"
    headers = {"Authorization": f"Bearer {WA_TOKEN}"}
    try:
        resp = requests.get(url, headers=headers, timeout=10)
        resp.raise_for_status()
        media_data = resp.json()
        file_url = media_data.get("url")
        mime_type = media_data.get("mime_type", "audio/ogg")
        # Очищаем mime_type (оставляем только 'audio/ogg', убирая 'codecs=opus')
        if ";" in mime_type:
            mime_type = mime_type.split(";")[0]

        file_resp = requests.get(file_url, headers=headers, timeout=20)
        file_resp.raise_for_status()
        return file_resp.content, mime_type
    except requests.RequestException as e:
        print(f"[WA MEDIA ERROR] {e}")
        return None, None


# --- ФУНКЦИЯ ОТПРАВКИ СООБЩЕНИЙ WHATSAPP ---
def send_whatsapp_template_message(phone_number: str, template_name: str, body_params: list[str]):
    url = f"https://graph.facebook.com/v18.0/{WA_PHONE_ID}/messages"
    headers = {
        "Authorization": f"Bearer {WA_TOKEN}",
        "Content-Type": "application/json",
    }
    payload = {
        "messaging_product": "whatsapp",
        "to": phone_number,
        "type": "template",
        "template": {
            "name": template_name,            # строго как в Business Manager
            "language": {"code": "ru"},       # или "en_US" / "ru_RU" и т.д.
            "components": [
                {
                    "type": "body",
                    "parameters": [
                        {"type": "text", "text": p} for p in body_params
                    ],
                }
            ],
        },
    }
    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=10)
        if resp.status_code >= 400:
            print(f"[WA ERROR] {resp.status_code} {resp.text}")
    except requests.RequestException as e:
        print(f"[WA EXCEPTION] {e}")


def send_whatsapp_to_owner(order_data):
    if not OWNER_PHONE_NUMBER:
        print("[OWNER_PHONE_NUMBER MISSING]")
        return

    # шаблон new_order_notification:
    # "Новый заказ: {{1}} (телефон {{2}}), материал {{3}}, цвет {{4}}, площадь {{5}} м², цена {{6}} тг, адрес {{7}}."
    title = order_data.get("name", "Не указано")
    details = (
        f"Телефон: {order_data.get('phone', 'Не указан')}, "
        f"материал: {order_data.get('material', 'Не указан')}, "
        f"цвет: {order_data.get('color', 'Не указан')}, "
        f"площадь: {order_data.get('area', 'Не указана')} м², "
        f"цена: {order_data.get('price', 'Не указана')} тг, "
        f"адрес: {order_data.get('address', 'Самовывоз')}."
    )
    params = [title, details]
    send_whatsapp_template_message(OWNER_PHONE_NUMBER, "new_order_notification", params)


def send_whatsapp_message(phone_number: str, text: str) -> None:
    url = f"https://graph.facebook.com/v18.0/{WA_PHONE_ID}/messages"
    headers = {
        "Authorization": f"Bearer {WA_TOKEN}",
        "Content-Type": "application/json",
    }
    payload = {
        "messaging_product": "whatsapp",
        "to": phone_number,
        "type": "text",
        "text": {"body": text},
    }
    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=10)
        if resp.status_code >= 400:
            print(f"[WA ERROR] {resp.status_code} {resp.text}")
    except requests.RequestException as e:
        print(f"[WA EXCEPTION] {e}")


# --- ЛОГИКА GEMINI (SYSTEM PROMPT & TOOLS) ---
# Текст SPIN-промпта — в rag/prompts.py: тот же промпт используют evals и CLI.

SYSTEM_PROMPT = build_bot_system_prompt(prices_text)

# Инициализация модели: инструменты агента (lookup_pricing, create_lead) и
# кэш моделей по системному промпту — в rag/agent.py.
model = get_model(SYSTEM_PROMPT)


def serialize_chat_history(chat) -> list[dict]:
    """
    Превращает историю Gemini ChatSession в JSON-совместимый список,
    чтобы можно было пережить рестарт сервера и восстановить диалог.

    Ограничение: части с function_call/function_response и бинарные данные
    (например, отправленное клиентом аудио) не сериализуются дословно — для
    аудио подставляется текстовая заглушка. Это осознанный компромисс: такие
    части встречаются либо в середине диалога (аудио, не критично для контекста
    следующих реплик), либо прямо перед закрытием сессии (create_lead),
    когда сессия и так удаляется.
    """
    serialized = []
    try:
        for content in chat.history:
            role = content.role
            text_parts = []
            for part in content.parts:
                text = getattr(part, "text", None)
                if isinstance(text, str) and text:
                    text_parts.append(text)
                elif getattr(part, "inline_data", None) is not None:
                    text_parts.append("[голосовое сообщение]")
            if text_parts:
                serialized.append({"role": role, "parts": text_parts})
    except Exception as e:
        print(f"[HISTORY SERIALIZE ERROR] {e}")
    return serialized


def build_chat_session(phone_number: str):
    """
    Создаёт ChatSession для номера, восстанавливая историю с диска, если она
    сохранена (например, после рестарта сервера). Вызывающий код должен уже
    держать state_lock или phone-lock — здесь дополнительной блокировки нет.
    """
    saved_history = open_leads.get(phone_number, {}).get("history", [])
    try:
        chat = model.start_chat(history=saved_history)
    except Exception as e:
        print(f"[HISTORY REBUILD ERROR] {phone_number}: {e}")
        chat = model.start_chat()
    chat_sessions[phone_number] = chat
    return chat


def process_gemini_response(phone_number, user_message=None, audio_data=None, mime_type=None):
    # --- ОБНОВЛЯЕМ ВРЕМЯ ПОСЛЕДНЕГО КОНТАКТА / ПОЛУЧАЕМ СЕССИЮ ---
    now = time.time()
    with state_lock:
        if phone_number in open_leads:
            # клиент снова проявился → разрешаем будущий follow-up
            open_leads[phone_number]["last_seen"] = now
            open_leads[phone_number]["followup_sent"] = False
        else:
            open_leads[phone_number] = {
                "last_seen": now,
                "followup_sent": False,
                "history": [],
            }
        save_leads(open_leads)

        chat = chat_sessions.get(phone_number)
        if chat is None:
            # либо совсем новый клиент, либо сервер перезапускался —
            # в обоих случаях build_chat_session корректно восстановит/создаст сессию
            chat = build_chat_session(phone_number)

    # --- АГЕНТ-ЦИКЛ (rag.agent): первый ход с контекстом базы знаний,
    # дальше инструменты lookup_pricing/create_lead до финального текста ---
    try:
        if user_message:
            log_chat_to_file(phone_number, "Клиент", user_message)
            result = run_agent(
                chat, phone_number,
                user_message=user_message, record_lead=_save_lead_to_sheets,
            )
        elif audio_data:
            log_chat_to_file(phone_number, "Клиент", "[Голосовое сообщение]")
            result = run_agent(
                chat, phone_number,
                audio_data=audio_data, mime_type=mime_type,
                record_lead=_save_lead_to_sheets,
            )
        else:
            return
    except Exception as e:
        print(f"[GEMINI ERROR] {phone_number}: {e}")
        send_whatsapp_message(
            phone_number,
            "Небольшая техническая заминка, попробуйте, пожалуйста, отправить сообщение ещё раз.",
        )
        return

    # Защита от пустого ответа
    if not result.text:
        send_whatsapp_message(
            phone_number,
            "Ошибка обработки запроса, попробуйте сформулировать вопрос иначе.",
        )
        return

    reply_text = result.text
    log_chat_to_file(phone_number, "ИИ-Бот", reply_text)
    send_whatsapp_message(phone_number, reply_text)

    # --- ЛИД СОХРАНЁН → ЗАКРЫВАЕМ СЕССИЮ; ИНАЧЕ СОХРАНЯЕМ ИСТОРИЮ ---
    with state_lock:
        if result.lead_saved:
            chat_sessions.pop(phone_number, None)
            open_leads.pop(phone_number, None)
            save_leads(open_leads)
        elif phone_number in open_leads:
            open_leads[phone_number]["history"] = serialize_chat_history(chat)
            save_leads(open_leads)


# --- ВЕБХУКИ WHATSAPP ---


@app.get("/")
async def root():
    return {"message": "Сервер работает! webhook находится на /webhook"}


@app.get("/webhook")
async def verify_webhook(
    hub_mode: str = Query(None, alias="hub.mode"),
    hub_challenge: str = Query(None, alias="hub.challenge"),
    hub_verify_token: str = Query(None, alias="hub.verify_token"),
):
    if hub_mode == "subscribe" and hub_verify_token == VERIFY_TOKEN:
        try:
            return int(hub_challenge)
        except (TypeError, ValueError):
            raise HTTPException(status_code=400, detail="Invalid hub.challenge")
    raise HTTPException(status_code=403, detail="Verification failed")


def verify_webhook_signature(raw_body: bytes, signature_header: str | None) -> bool:
    """
    Проверяет подпись X-Hub-Signature-256, которую Meta добавляет к каждому
    вебхуку. Без этой проверки любой человек, узнавший URL вебхука, может
    слать поддельные "сообщения от клиента" и триггерить создание лидов.
    """
    if not WA_APP_SECRET:
        # Секрет не настроен — не можем проверить подпись. Не блокируем работу бота
        # (чтобы не сломать существующий деплой "из коробки"), но громко предупреждаем.
        print(
            "[SECURITY WARNING] WA_APP_SECRET не задан в .env — подпись вебхука "
            "НЕ проверяется, вебхук уязвим для подделки запросов. "
            "Возьмите App Secret в Meta App Dashboard -> Settings -> Basic."
        )
        return True

    if not signature_header or not signature_header.startswith("sha256="):
        return False

    expected = hmac.new(WA_APP_SECRET.encode("utf-8"), raw_body, hashlib.sha256).hexdigest()
    provided = signature_header.split("=", 1)[1]
    return hmac.compare_digest(expected, provided)


def handle_single_message(msg: dict) -> None:
    """
    Синхронная обработка одного входящего сообщения. Выполняется в отдельном
    потоке (через asyncio.to_thread), чтобы не блокировать event loop FastAPI —
    иначе, пока идёт обращение к Gemini/WhatsApp API для одного клиента, ВСЕ
    остальные клиенты ждут своей очереди.
    """
    try:
        phone_number = msg["from"]
        msg_type = msg.get("type")

        # Все сообщения одного номера обрабатываются строго последовательно
        with phone_lock(phone_number):
            if msg_type == "text":
                text = msg["text"]["body"]
                process_gemini_response(phone_number, user_message=text)

            elif msg_type == "audio":
                media_id = msg["audio"]["id"]
                audio_bytes, mime_type = download_whatsapp_media(media_id)
                if audio_bytes:
                    process_gemini_response(
                        phone_number,
                        audio_data=audio_bytes,
                        mime_type=mime_type,
                    )
                else:
                    send_whatsapp_message(
                        phone_number,
                        "Не удалось загрузить ваше голосовое сообщение. Напишите, пожалуйста, текстом.",
                    )
            else:
                send_whatsapp_message(
                    phone_number,
                    "Пока я умею работать только с текстом и голосовыми сообщениями.",
                )
    except Exception as e:
        print(f"[MESSAGE HANDLING ERROR] {e}")


@app.post("/webhook")
async def receive_webhook(req: Request):
    raw_body = await req.body()
    signature = req.headers.get("X-Hub-Signature-256")

    if not verify_webhook_signature(raw_body, signature):
        raise HTTPException(status_code=403, detail="Invalid signature")

    try:
        body = json.loads(raw_body)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON body")

    messages_to_process = []
    try:
        if body.get("entry"):
            for entry in body["entry"]:
                for change in entry.get("changes", []):
                    if change.get("field") == "messages":
                        for msg in change["value"].get("messages", []):
                            # is_duplicate_message пишет seen_messages на диск —
                            # синхронный I/O нельзя держать на event loop
                            if await asyncio.to_thread(is_duplicate_message, msg.get("id")):
                                print(
                                    f"[DUPLICATE MESSAGE] {msg.get('id')} — "
                                    "повторная доставка вебхука, пропускаем"
                                )
                                continue
                            messages_to_process.append(msg)
    except Exception as e:
        print(f"[WEBHOOK PARSE ERROR] {e}")

    # Запускаем обработку каждого сообщения в отдельном потоке и СРАЗУ отвечаем
    # WhatsApp "ok". Раньше сервер ждал полного ответа от Gemini/Sheets, прежде
    # чем ответить на вебхук — если это занимало больше нескольких секунд,
    # WhatsApp мог посчитать доставку неуспешной и повторно прислать тот же вебхук,
    # что приводило к двойной обработке одного сообщения.
    for msg in messages_to_process:
        asyncio.create_task(asyncio.to_thread(handle_single_message, msg))

    return {"status": "ok"}


def check_stale_leads():
    """Проверяет клиентов, которые не отвечали больше 4 часов и отправляет ОДНО напоминание"""
    STALE_TIME = 4 * 60 * 60  # 4 часа в секундах
    # Если клиент не ответил даже на follow-up за это время — считаем лида
    # окончательно остывшим и перестаём хранить его состояние, чтобы open_leads
    # и chat_sessions не росли бесконечно.
    MAX_LEAD_AGE = 48 * 60 * 60  # 48 часов

    # Снимок ключей под state_lock: open_leads пишется/чистится другими потоками
    # (process_gemini_response), итерировать словарь без лока нельзя.
    with state_lock:
        phones = list(open_leads.keys())

    for phone in phones:
        with state_lock:
            lead = open_leads.get(phone)
            if lead is None:
                continue
            last_seen = lead.get("last_seen", time.time())
            elapsed_time = time.time() - last_seen
            followup_sent = lead.get("followup_sent", False)

        if elapsed_time > MAX_LEAD_AGE:
            with state_lock:
                chat_sessions.pop(phone, None)
                open_leads.pop(phone, None)
                save_leads(open_leads)
            continue

        # Если клиент молчит дольше порога и follow-up ещё НЕ отправляли
        if elapsed_time > STALE_TIME and not followup_sent:
            with phone_lock(phone):
                # Между снимком и локом клиент мог написать (handle_single_message
                # берёт тот же лок) — состояние проверяем заново уже под локом
                with state_lock:
                    lead = open_leads.get(phone)
                    if lead is None:
                        continue
                    if time.time() - lead.get("last_seen", time.time()) <= STALE_TIME:
                        continue
                    if lead.get("followup_sent", False):
                        continue
                    chat = chat_sessions.get(phone)
                    if chat is None:
                        # Сессии в памяти нет (например, сервер перезапускался) —
                        # восстанавливаем её из сохранённой на диске истории,
                        # вместо того чтобы молча терять лида.
                        chat = build_chat_session(phone)

                prompt = (
                    "СИСТЕМНОЕ СООБЩЕНИЕ: Клиент не отвечает уже более 4 часов. "
                    "Напиши ОДНО очень короткое, ненавязчивое сообщение, чтобы мягко вернуть его в диалог. "
                    "Используй контекст нашей беседы. Не задавай прямых вопросов 'вы тут?', лучше напомни о чем-то важном "
                    "по объекту или предложи помощь. Без эмодзи."
                )

                try:
                    gemini_rate_limiter.acquire()
                    response = chat.send_message(prompt)
                    # у response.text дефолт "" уже есть — getattr с фолбэком не
                    # сработал бы; явно проверяем пустоту
                    reply_text = (getattr(response, "text", "") or "").strip()
                    if not reply_text:
                        reply_text = "Если что, я здесь и могу помочь по вашему объекту."

                    send_whatsapp_message(phone, reply_text)
                    log_chat_to_file(phone, "ИИ-Бот (Follow-up)", reply_text)

                    with state_lock:
                        if phone in open_leads:
                            # ПОМЕТИМ, ЧТО FOLLOW-UP УЖЕ ОТПРАВЛЕН, но НЕ обновляем last_seen,
                            # чтобы видеть реальный возраст лида
                            open_leads[phone]["followup_sent"] = True
                            open_leads[phone]["history"] = serialize_chat_history(chat)
                            save_leads(open_leads)
                except Exception as e:
                    print(f"Ошибка при отправке Follow-up для {phone}: {e}")


# Создаем планировщик (будет запущен вместе с приложением)
scheduler = BackgroundScheduler()


@app.on_event("startup")
def start_scheduler():
    # Проверяем каждые 10 минут (можете изменить на minutes=1 для теста)
    scheduler.add_job(check_stale_leads, "interval", minutes=10)
    scheduler.start()


@app.on_event("shutdown")
def shutdown_scheduler():
    scheduler.shutdown()