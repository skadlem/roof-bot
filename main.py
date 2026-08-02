import os
import json
import time
import hmac
import hashlib
import asyncio
import threading
from collections import deque
import requests
import gspread
import google.generativeai as genai
from fastapi import FastAPI, Request, HTTPException, Query
from dotenv import load_dotenv
from datetime import datetime

from apscheduler.schedulers.background import BackgroundScheduler

LEADS_FILE = "open_leads.json"

# Реентрантная блокировка — защищает open_leads И chat_sessions от гонок,
# когда несколько сообщений (разных клиентов) обрабатываются параллельно в разных потоках.
state_lock = threading.RLock()

# Отдельные локи НА КАЖДЫЙ номер телефона — гарантируют, что два сообщения
# от ОДНОГО И ТОГО ЖЕ клиента обрабатываются строго по очереди, а не параллельно
# (иначе можно словить гонку внутри одного Gemini ChatSession).
_phone_locks_guard = threading.Lock()
_phone_locks: dict[str, threading.Lock] = {}


def get_phone_lock(phone_number: str) -> threading.Lock:
    with _phone_locks_guard:
        if phone_number not in _phone_locks:
            _phone_locks[phone_number] = threading.Lock()
        return _phone_locks[phone_number]


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
# ответа, а в create_order может уйти задвоенный заказ. Храним ID уже обработанных
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


def load_prices():
    try:
        with open("prices.json", "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"Не удалось загрузить prices.json: {e}")
        return {}


# Загружаем цены при старте
PRICES = load_prices()
prices_text = "\n".join([f"- {k}: {v} тг за кв.м" for k, v in PRICES.items()])

# --- БАЗА ЗНАНИЙ (RAG) ---
# kb_embeddings.json собирается скриптом build_kb.py из kb/*.md.
# Файла нет или он пуст → KB = None, поиск фактов отключён, бот не падает.
KB_EMBEDDINGS_FILE = "kb_embeddings.json"
KB_SIMILARITY_THRESHOLD = 0.3
KB_TOP_K = 3
KB = None


def load_kb():
    global KB
    try:
        with open(KB_EMBEDDINGS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not data.get("chunks") or not data.get("vectors"):
            return None
        KB = data
    except Exception as e:
        print(f"Не удалось загрузить {KB_EMBEDDINGS_FILE}: {e}")
        KB = None
    return KB


def _cosine_similarity(a, b):
    if len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(y * y for y in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def retrieve_kb(query, k=KB_TOP_K):
    """Возвращает до k фактов базы знаний, близких к запросу, или ""."""
    if KB is None:
        return ""
    try:
        result = genai.embed_content(
            model="models/gemini-embedding-001",
            content=query,
            task_type="RETRIEVAL_QUERY",
        )
        query_vec = result["embedding"]
    except Exception as e:
        print(f"[KB EMBED ERROR] {e}")
        return ""
    scored = sorted(
        ((_cosine_similarity(query_vec, v), i) for i, v in enumerate(KB["vectors"])),
        reverse=True,
    )
    hits = []
    for score, i in scored:
        if score < KB_SIMILARITY_THRESHOLD:
            break
        hits.append(KB["chunks"][i])
        if len(hits) >= k:
            break
    return "\n\n".join(hits)


def build_message_with_kb(user_message):
    """Подставляет факты базы знаний в сообщение клиента как контекст для Gemini."""
    context = retrieve_kb(user_message)
    if not context:
        return user_message
    return (
        "БАЗА ЗНАНИЙ КОМПАНИИ (не цитируй дословно, используй только эти факты):\n"
        f"{context}\n\n{user_message}"
    )


app = FastAPI()

# --- ИНИЦИАЛИЗАЦИЯ ---
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

WA_TOKEN = os.getenv("WA_TOKEN")
WA_PHONE_ID = os.getenv("WA_PHONE_ID")
VERIFY_TOKEN = os.getenv("WA_VERIFY_TOKEN")
OWNER_PHONE_NUMBER = os.getenv("OWNER_PHONE_NUMBER")

# Секрет приложения Meta (App Dashboard -> Settings -> Basic -> App Secret).
# ЭТО НЕ WA_TOKEN. Нужен для проверки подписи входящих вебхуков.
WA_APP_SECRET = os.getenv("WA_APP_SECRET")

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
            _sheet = gc.open_by_key("11vKc3-d5zhX1-0wnua3blinCh4R6RJBi555JOhHz218").sheet1
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
    except Exception as e:
        print(f"[GOOGLE SHEETS ERROR] {e}")


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
            print(f"[WA TEMPLATE ERROR] {resp.status_code} {resp.text}")
    except requests.RequestException as e:
        print(f"[WA TEMPLATE EXCEPTION] {e}")


def send_whatsapp_to_owner(order_data):
    if not OWNER_PHONE_NUMBER:
        print("[OWNER_PHONE_NUMBER MISSING]")
        return

    # допустим, шаблон new_order_notification выглядит как:
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


# --- ЛОГИКА GEMINI (SYSTEM PROMPT & FUNCTIONS) ---

SYSTEM_PROMPT = f"""
Ты — топовый эксперт и менеджер по продажам компании "МеталлКровля". Клиент может отправлять голосовые сообщения. 
Твоя цель — провести клиента по технике SPIN-продаж: диагностика → раскрытие боли → презентация-зеркало → мягкая отработка → закрытие на маленький шаг.

ПРАВИЛО ПЕРВОГО СООБЩЕНИЯ:
- Если это ПЕРВОЕ твое сообщение в диалоге, начни с очень короткого представления:
  "Вы обратились в компанию МеталлКровля, я виртуальный менеджер по кровле."
- После этой одной фразы сразу задай первый диагностический вопрос.
- ВО ВСЕХ последующих сообщениях не представляйся повторно.

СТИЛЬ ОБЩЕНИЯ (СТРОГО):
- Пиши КАК В МЕССЕНДЖЕРЕ. 1-3 предложения максимум.
- В ОДНОМ сообщении СТРОГО ОДИН вопрос. Никогда не задавай несколько вопросов сразу.
- ЗАПРЕЩЕНО использовать списки, буллиты, абзацы и эмодзи. Только простой короткий текст.
- Обосновывай вопросы ("спрашиваю, потому что...") — это выглядит как экспертиза, а не допрос.
- Никогда не имитируй искусственную срочность ("акция только сегодня").

ПРАЙС-ЛИСТ (цена за 1 кв.м):
{prices_text}

БАЗА ЗНАНИЙ:
Отвечай на фактические вопросы клиента (услуги, сроки, доставка, гарантии) строго на основе прайс-листа выше и контекста из базы знаний, который приходит в сообщении. НЕ выдумывай цены, услуги, сроки, зоны доставки или гарантии. Если клиент спрашивает факт, которого нет ни в прайсе, ни в контексте базы знаний, скажи, что уточнишь у менеджера, и предложи передать контакт менеджера. Никогда не называй выдуманные цифры.

АЛГОРИТМ РАБОТЫ:

ЭТАП 1: ДИАГНОСТИКА (Снимает ощущение впаривания)
Никогда не начинай с цены. Задавай вопросы по цепочке (по одному за раз):
1. Что за объект (дом/коммерция/промышленное)?
2. Этап стройки (коробка без крыши / меняете старую / течет ремонт)?
3. Площадь и примерная конфигурация (скаты, сложная геометрия)?
4. Сроки (есть ли дедлайн, когда нужно закрыть)?

ЭТАП 2: РАСКРЫТИЕ БОЛИ (Клиент продает себе сам)
Узнай, что уже не устроило у других. Спроси: "Скорее всего вы уже с парой кровельщиков пообщались? Что смущало — цена, сроки, или непонятность?"
Клиент назовет боли (например: боязь, что цена вырастет, или недоверие). ЗАПОМНИ ИХ.

ЭТАП 3: ПРЕЗЕНТАЦИЯ-ЗЕРКАЛО
Используй ТОЛЬКО те слова и боли, которые назвал клиент. 
Пример: "Смотрите, вы сказали, что для вас важно, чтобы итоговая сумма не выросла. Как раз под это мы фиксируем смету до начала работ."
После презентации узнай имя клиента, адрес доставки и рассчитай ИТОГОВУЮ стоимость (Цена за кв.м * Площадь). Назови сумму.

ЭТАП 4: ОТРАБОТКА ВОЗРАЖЕНИЙ (Филигранно)
Формула: Согласие → Уточняющий вопрос → Точечный контраргумент → Маленький шаг.
- Если "Дорого": "Да, сумма ощутимая. Дорого по сравнению с другим предложением, или просто бюджет тянет?" (Если сравнивает: "Окей, часто разница из-за того, что туда не включена гидроизоляция. Давайте я подготовлю смету с разбивкой, чтобы честно сравнить?").
- Если "Подумаю": "Конечно, дело серьезное. Думать будете над суммой, сроками, или хочется сравнить?" 
Не дави. Предлагай маленькие шаги (скинуть смету, разбить на этапы).

ЭТАП 5: ЗАВЕРШЕНИЕ
- Если клиент СОГЛАСЕН на маленький шаг или подтвердил заказ → вызови функцию `create_order`, передав ИТОГОВУЮ ЦЕНУ, имя, адрес, материал, цвет и площадь.
- Если клиент ОКОНЧАТЕЛЬНО отказался после 2-3 твоих попыток отработать возражение → вежливо попрощайся ("Оставлю смету у вас, если что — обращайтесь") → вызови функцию `end_chat`.

Данные, которые обязательно нужно собрать для функции create_order: Имя, Материал, Цвет, Площадь, Итоговая стоимость, Адрес доставки.
"""

# Функция 1: Создание заказа (добавлена цена)
create_order_function = {
    "name": "create_order",
    "description": "Вызови эту функцию, когда клиент согласился и подтвердил заказ. Обязательно посчитай итоговую цену.",
    "parameters": {
        "type": "object",
        "properties": {
            "name": {"type": "string", "description": "Имя клиента"},
            "material": {
                "type": "string",
                "description": "Тип материала (металлочерепица, профнастил и т.д.)",
            },
            "color": {"type": "string", "description": "Цвет материала"},
            "area": {
                "type": "string",
                "description": "Площадь в квадратных метрах",
            },
            "price": {
                "type": "string",
                "description": "Итоговая расчетная стоимость заказа в тенге (только цифры)",
            },
            "address": {
                "type": "string",
                "description": "Адрес доставки или 'Самовывоз'",
            },
        },
        "required": ["name", "material", "color", "area", "price"],
    },
}

# Функция 2: Завершение чата (если клиент ушел)
end_chat_function = {
    "name": "end_chat",
    "description": "Вызови эту функцию, если клиент окончательно отказался от покупки и разговор завершен",
    "parameters": {
        "type": "object",
        "properties": {},
    },
}

# Инициализация модели
model = genai.GenerativeModel(
    model_name="models/gemini-3.1-flash-lite",
    system_instruction=SYSTEM_PROMPT,
    tools=[{"function_declarations": [create_order_function, end_chat_function]}],
)


class RateLimiter:
    """
    Потокобезопасный sliding-window лимитер запросов. Лимит подписки Gemini API
    (15 запросов/мин) действует на весь ключ/проект целиком, а не на клиента —
    поэтому лимитер один общий на все потоки, а не per-phone.

    acquire() блокирует вызывающий поток до тех пор, пока не появится свободный
    слот. Так как Gemini-запросы и так выполняются в отдельных потоках (через
    asyncio.to_thread), это ожидание не блокирует event loop FastAPI — другие
    клиенты продолжают обслуживаться, просто их Gemini-запросы тоже встанут
    в общую очередь лимитера.
    """

    def __init__(self, max_calls: int, period_seconds: float):
        self.max_calls = max_calls
        self.period = period_seconds
        self.calls: deque[float] = deque()
        self.lock = threading.Lock()

    def acquire(self):
        while True:
            with self.lock:
                now = time.time()
                while self.calls and now - self.calls[0] > self.period:
                    self.calls.popleft()

                if len(self.calls) < self.max_calls:
                    self.calls.append(now)
                    return

                wait_time = self.period - (now - self.calls[0]) + 0.05
            time.sleep(max(wait_time, 0.05))


# Лимит тарифа Gemini API — 15 запросов/мин на весь ключ/проект; держим запас
# 14 вместо 15 на случай погрешности таймингов. Переопределяется в .env (GEMINI_RPM).
GEMINI_RPM = int(os.getenv("GEMINI_RPM", "14"))
gemini_rate_limiter = RateLimiter(max_calls=GEMINI_RPM, period_seconds=60)


def serialize_chat_history(chat) -> list[dict]:
    """
    Превращает историю Gemini ChatSession в JSON-совместимый список,
    чтобы можно было пережить рестарт сервера и восстановить диалог.

    Ограничение: части с function_call/function_response и бинарные данные
    (например, отправленное клиентом аудио) не сериализуются дословно — для
    аудио подставляется текстовая заглушка. Это осознанный компромисс: такие
    части встречаются либо в середине диалога (аудио, не критично для контекста
    следующих реплик), либо прямо перед закрытием сессии (create_order/end_chat),
    когда сессия и так удаляется.
    """
    serialized = []
    try:
        for content in chat.history:
            role = content.role
            text_parts = []
            for part in content.parts:
                text = getattr(part, "text", None)
                if text:
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

    if user_message:
        log_chat_to_file(phone_number, "Клиент", user_message)
        user_message = build_message_with_kb(user_message)
        try:
            gemini_rate_limiter.acquire()
            response = chat.send_message(user_message)
        except Exception as e:
            print(f"[GEMINI ERROR] {phone_number}: {e}")
            send_whatsapp_message(
                phone_number,
                "Небольшая техническая заминка, попробуйте, пожалуйста, отправить сообщение ещё раз.",
            )
            return
    elif audio_data:
        log_chat_to_file(phone_number, "Клиент", "[Голосовое сообщение]")
        try:
            gemini_rate_limiter.acquire()
            response = chat.send_message([{"mime_type": mime_type, "data": audio_data}])
        except Exception as e:
            print(f"[GEMINI ERROR] {phone_number}: {e}")
            send_whatsapp_message(
                phone_number,
                "Небольшая техническая заминка, попробуйте, пожалуйста, отправить сообщение ещё раз.",
            )
            return
    else:
        return

    # Защита от пустого ответа
    if not getattr(response, "candidates", None):
        send_whatsapp_message(
            phone_number,
            "Ошибка обработки запроса, попробуйте сформулировать вопрос иначе.",
        )
        return

    candidate = response.candidates[0]
    content = getattr(candidate, "content", None)
    parts = getattr(content, "parts", []) if content else []

    should_end_session = False

    for part in parts:
        if hasattr(part, "function_call") and part.function_call.name:
            func_name = part.function_call.name

            if func_name == "create_order":
                args = part.function_call.args
                order_data = {
                    "name": args.get("name", ""),
                    "material": args.get("material", ""),
                    "color": args.get("color", ""),
                    "area": args.get("area", ""),
                    "price": args.get("price", "0"),
                    "address": args.get("address", "Самовывоз"),
                    "phone": phone_number,
                }
                send_whatsapp_to_owner(order_data)
                add_to_google_sheets(order_data)

                try:
                    gemini_rate_limiter.acquire()
                    response = chat.send_message(
                        genai.protos.Content(
                            parts=[
                                genai.protos.Part(
                                    function_response=genai.protos.FunctionResponse(
                                        name="create_order",
                                        response={"result": "success"},
                                    )
                                )
                            ]
                        )
                    )
                except Exception as e:
                    print(f"[GEMINI ERROR] {phone_number}: {e}")
                should_end_session = True
                break

            elif func_name == "end_chat":
                try:
                    gemini_rate_limiter.acquire()
                    response = chat.send_message(
                        genai.protos.Content(
                            parts=[
                                genai.protos.Part(
                                    function_response=genai.protos.FunctionResponse(
                                        name="end_chat",
                                        response={"result": "success"},
                                    )
                                )
                            ]
                        )
                    )
                except Exception as e:
                    print(f"[GEMINI ERROR] {phone_number}: {e}")
                should_end_session = True
                break

    reply_text = getattr(response, "text", None)
    if not reply_text:
        reply_text = (
            "Не удалось сформировать ответ, давайте попробуем переформулировать вопрос."
        )

    log_chat_to_file(phone_number, "ИИ-Бот", reply_text)
    send_whatsapp_message(phone_number, reply_text)

    # --- УДАЛЯЕМ ЛИДА ПРИ ЗАВЕРШЕНИИ ИЛИ СОХРАНЯЕМ ИСТОРИЮ ---
    with state_lock:
        if should_end_session:
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
    слать поддельные "сообщения от клиента" и триггерить create_order/оповещения.
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
        with get_phone_lock(phone_number):
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
                            if is_duplicate_message(msg.get("id")):
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
    current_time = time.time()
    STALE_TIME = 4 * 60 * 60  # 4 часа в секундах
    # Если клиент не ответил даже на follow-up за это время — считаем лида
    # окончательно остывшим и перестаём хранить его состояние, чтобы open_leads
    # и chat_sessions не росли бесконечно.
    MAX_LEAD_AGE = 48 * 60 * 60  # 48 часов

    # Идем по копии словаря, чтобы можно было удалять элементы во время итерации
    for phone, lead in list(open_leads.items()):
        last_seen = lead.get("last_seen", current_time)
        followup_sent = lead.get("followup_sent", False)
        elapsed_time = current_time - last_seen

        if elapsed_time > MAX_LEAD_AGE:
            with state_lock:
                chat_sessions.pop(phone, None)
                open_leads.pop(phone, None)
                save_leads(open_leads)
            continue

        # Если клиент молчит дольше порога и follow-up ещё НЕ отправляли
        if elapsed_time > STALE_TIME and not followup_sent:
            with get_phone_lock(phone):
                with state_lock:
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
                    reply_text = getattr(
                        response,
                        "text",
                        "Если что, я здесь и могу помочь по вашему объекту.",
                    )

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