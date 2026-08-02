"""Агент-цикл Gemini: диалог с инструментами (lookup_pricing, create_lead).

Первый ход сообщение клиента оборачивается контекстом RAG-базы (шаг 2:
rag.retrieve + rag.prompts). Дальше цикл: ответ модели с function_call →
выполнение инструмента → результат обратно в сессию → следующий вызов —
пока модель не ответит текстом или не исчерпает MAX_TURNS обращений к Gemini.

Используется ботом (main.py — модель через get_model(SYSTEM_PROMPT), свой
промпт SPIN-продаж) и CLI (rag/agent_cli.py — DEFAULT_SYSTEM_PROMPT).
Бот вызывает run_agent под per-phone lock (main.handle_single_message),
так что параллельных вызовов на одну сессию нет.
"""

import json
import os
import re
import threading
import time
from collections import deque
from dataclasses import dataclass, field

import google.generativeai as genai

from rag import retrieve as rag_retrieve
from rag.prompts import build_rag_message

# та же модель, что у бота (main.py) и rag/ask.py
MODEL_NAME = "models/gemini-3.1-flash-lite"

# Сколько чанков базы знаний подмешивается в первый ход (было KB_TOP_K в main.py)
KB_TOP_K = 3

# Максимум обращений к Gemini за одно сообщение клиента: первый ход с контекстом
# + до 2 ходов после ответов инструментов. Дальше считаем, что модель зациклилась.
MAX_TURNS = 3

PRICES_FILE = "prices.json"


class RateLimiter:
    """Потокобезопасный sliding-window лимитер запросов (один на все потоки:
    лимит Gemini действует на ключ целиком, а не на клиента). acquire()
    блокирует вызывающий поток, пока не освободится слот."""

    def __init__(self, max_calls, period_seconds):
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


def load_prices():
    try:
        with open(PRICES_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"Не удалось загрузить prices.json: {e}")
        return {}


# Промпт для CLI (rag/agent_cli.py). У бота свой системный промпт (SPIN-продажи),
# инструменты те же.
DEFAULT_SYSTEM_PROMPT = """\
Ты — менеджер по продажам компании «МеталлКровля» (кровельные работы).
Отвечай кратко, как в мессенджере (1-3 предложения).

У тебя есть два инструмента:
- lookup_pricing(service) — точная цена услуги из прайс-листа. Цены называй ТОЛЬКО через этот инструмент (или из контекста базы знаний, если он приложен к сообщению).
- create_lead(name, phone, service, message) — сохранить лид клиента, когда он готов оставить заявку или подтвердил заказ. Телефон можно не указывать — подставится номер клиента автоматически. Имя обязательно: нет имени — попроси клиента уточнить, функцию с пустыми полями не вызывай.

На фактические вопросы клиента (услуги, цены, скидки, сроки, доставка,
гарантии) отвечай строго на основе контекста базы знаний из сообщения и
ответов инструментов. НЕ выдумывай цифры, скидки, сроки или факты: нет
информации ни в контексте, ни в ответах инструментов — скажи «уточню у
менеджера» и предложи передать вопрос менеджеру."""


# Декларации функций в формате легаси google-generativeai (0.8.x): как было
# с create_order/end_chat в main.py, только нашими двумя инструментами.
TOOLS = [
    {
        "function_declarations": [
            {
                "name": "lookup_pricing",
                "description": (
                    "Цена услуги за кв.м из прайс-листа компании. Вызывай, "
                    "когда клиент спрашивает цену или стоимость услуги."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "service": {
                            "type": "string",
                            "description": (
                                "Название услуги/материала (металлочерепица, "
                                "профнастил, фальц, мягкая кровля и т.п.)"
                            ),
                        },
                    },
                    "required": ["service"],
                },
            },
            {
                "name": "create_lead",
                "description": (
                    "Сохранить лид клиента: имя, телефон, услуга, сообщение. "
                    "Вызывай, когда клиент готов оставить заявку или подтвердил заказ. "
                    "Телефон можно не указывать — подставится номер клиента автоматически."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "description": "Имя клиента"},
                        "phone": {
                            "type": "string",
                            "description": (
                                "Телефон клиента в международном формате. "
                                "Можно не указывать — подставится номер клиента автоматически"
                            ),
                        },
                        "service": {
                            "type": "string",
                            "description": "Услуга/материал, который интересует клиента",
                        },
                        "message": {
                            "type": "string",
                            "description": "Краткое описание запроса клиента",
                        },
                    },
                    "required": ["name", "service", "message"],
                },
            },
        ]
    }
]


# Кэш моделей по системному промпту: у бота свой (SPIN-продажи), у CLI —
# DEFAULT_SYSTEM_PROMPT. genai.configure должен быть вызван до первого вызова.
_models: dict[str, genai.GenerativeModel] = {}


def get_model(system_prompt=None):
    """Модель с инструментами агента; system_prompt=None → DEFAULT_SYSTEM_PROMPT."""
    key = system_prompt or DEFAULT_SYSTEM_PROMPT
    if key not in _models:
        _models[key] = genai.GenerativeModel(
            model_name=MODEL_NAME, system_instruction=key, tools=TOOLS
        )
    return _models[key]


# --- ИНСТРУМЕНТЫ ---


def lookup_pricing(service):
    """Цена услуги за кв.м из prices.json — живого прайса бота (main.py вставляет
    его же в системный промпт; kb/faq.md содержит устаревшую копию).

    Прайс читается на каждый вызов, чтобы правки файла применялись сразу.
    Сначала точное совпадение ключа, затем частичное вхождение по имени.
    """
    service = (service or "").strip().lower()
    if not service:
        return "Ошибка: не указана услуга. Уточни у клиента, какая услуга его интересует."
    prices = load_prices()
    for name, price in prices.items():
        if service == name.lower() or service in name.lower() or name.lower() in service:
            return f"{name}: {price} тг за кв.м"
    available = ", ".join(prices) or "прайс пуст"
    return (
        f"Услуга «{service}» не найдена в прайсе. Доступно: {available}. "
        "Скажи клиенту, что уточнишь цену у менеджера."
    )


def _phone_digits(phone):
    return re.sub(r"\D", "", phone or "")


def create_lead(name, phone, service, message, *, default_phone=None, record_lead=None):
    """Сохраняет лид в Google Sheets через переданный хук record_lead(order_data).

    Валидация на входе: невалидно — возвращается текст ошибки, который модель
    передаёт клиенту (исключения не бросаются). record_lead — в боте это
    main.add_to_google_sheets, в CLI — заглушка.
    """
    name = (name or "").strip()
    phone = (phone or "").strip() or default_phone
    if not name:
        return "Ошибка: не указано имя клиента. Спроси, как зовут клиента."
    digits = _phone_digits(phone)
    if not 7 <= len(digits) <= 15:
        return (
            f"Ошибка: «{phone or ''}» не похож на номер телефона. "
            "Попроси клиента прислать номер в международном формате."
        )
    order_data = {
        "name": name,
        "phone": phone,
        "material": (service or "").strip(),
        "color": "",
        "area": "",
        "price": "",
        # текст запроса — в свободную текстовую колонку «Адрес»
        "address": (message or "").strip() or "Самовывоз",
    }
    if record_lead:
        if record_lead(order_data) is False:
            # хук вернул явный False (таблица недоступна) — клиенту уходит
            # ошибка, сессия не закрывается, лид можно повторить.
            # None (например, list.append) считается успехом.
            return "Ошибка: не удалось сохранить заявку. Попробуйте отправить её ещё раз."
    text = f"Лид сохранён: {name}, {phone}"
    if order_data["material"]:
        text += f", услуга: {order_data['material']}"
    return text + ". Поблагодари клиента и скажи, что менеджер свяжется с ним."


def _execute_tool(name, args, client_id, record_lead):
    """Выполняет инструмент по имени; возвращает dict для FunctionResponse."""
    if name == "lookup_pricing":
        return {"result": lookup_pricing(args.get("service"))}
    if name == "create_lead":
        return {
            "result": create_lead(
                args.get("name"), args.get("phone"), args.get("service"),
                args.get("message"), default_phone=client_id, record_lead=record_lead,
            )
        }
    return {"result": f"Ошибка: неизвестный инструмент «{name}»."}


def _execute_calls(calls, client_id, record_lead, result):
    """Выполняет function_calls и возвращает parts FunctionResponse для сессии.

    Дубль create_lead в одном цикле не выполняется (result.lead_saved уже
    True): заявка в таблице, повторный вызов задвоил бы лида. Об успехе
    судим по тексту инструмента: ошибки валидации начинаются с «Ошибка:»
    и не закрывают сессию.
    """
    feedback = []
    for call in calls:
        args = call.args or {}
        if call.name == "create_lead" and result.lead_saved:
            outcome = {
                "result": (
                    "Заявка уже сохранена ранее. Поблагодари клиента и скажи, "
                    "что менеджер свяжется с ним."
                )
            }
        else:
            outcome = _execute_tool(call.name, args, client_id, record_lead)
        if call.name == "create_lead" and not outcome["result"].startswith("Ошибка:"):
            result.lead_saved = True
        result.tool_calls.append((call.name, args, outcome["result"]))
        feedback.append(
            genai.protos.Part(
                function_response=genai.protos.FunctionResponse(
                    name=call.name, response=outcome,
                )
            )
        )
    return feedback


# --- АГЕНТ-ЦИКЛ ---


@dataclass
class AgentResult:
    """Итог обработки одного сообщения клиента."""
    text: str = ""              # финальный ответ модели (пусто — ответа нет)
    lead_saved: bool = False    # create_lead выполнен успешно → сессию закрываем
    tool_calls: list = field(default_factory=list)  # (имя, args, результат) — для CLI
    hits: list = None           # чанки базы знаний первого хода — для CLI


def _content_parts(response):
    candidate = getattr(response, "candidates", None)
    if not candidate:
        return []
    content = getattr(candidate[0], "content", None)
    return getattr(content, "parts", []) if content else []


def run_agent(chat, client_id, *, user_message=None, audio_data=None,
              mime_type=None, record_lead=None, kb_top_k=KB_TOP_K):
    """Полный агент-цикл для одного сообщения клиента в сессии chat.

    Первый ход: текст оборачивается контекстом базы знаний (то, что делал
    build_message_with_kb в шаге 2), аудио идёт сырыми parts. Дальше до
    MAX_TURNS обращений к Gemini: ответ с function_call → выполнение
    инструмента → результат обратно в сессию → следующий ход.

    Вызывается под per-phone lock бота (main.handle_single_message), поэтому
    параллельных вызовов на одну сессию нет; лимитер Gemini общий на все потоки.
    """
    if audio_data is not None:
        first_message = [{"mime_type": mime_type, "data": audio_data}]
        hits = None
    else:
        try:
            hits = rag_retrieve.retrieve(user_message, client_id, k=kb_top_k)
            first_message = build_rag_message(user_message, hits)
        except Exception as e:
            print(f"[RAG RETRIEVE ERROR] {client_id}: {e}")
            first_message, hits = user_message, None

    result = AgentResult(hits=hits)
    gemini_rate_limiter.acquire()
    response = chat.send_message(first_message)

    for _ in range(MAX_TURNS - 1):
        parts = _content_parts(response)
        calls = [
            p.function_call for p in parts
            if getattr(p, "function_call", None) and p.function_call.name
        ]
        if not calls:
            break
        feedback = _execute_calls(calls, client_id, record_lead, result)
        gemini_rate_limiter.acquire()
        response = chat.send_message(genai.protos.Content(parts=feedback))

    # Цикл мог исчерпать MAX_TURNS, когда в последнем ответе модели остались
    # невыполненные function_calls (например, create_lead в самом конце) —
    # выполняем их без нового обращения к Gemini, чтобы заявка не потерялась.
    # В сессию результат не отправляем: обращений больше не будет, а лишний
    # user-контент в истории ничем не помогает.
    trailing_calls = [
        p.function_call for p in _content_parts(response)
        if getattr(p, "function_call", None) and p.function_call.name
    ]
    if trailing_calls:
        _execute_calls(trailing_calls, client_id, record_lead, result)

    result.text = getattr(response, "text", None) or ""
    # Лид сохранён, а текст не пришёл (модель позвала create_lead последним
    # ходом и цикл исчерпал MAX_TURNS) — клиенту нужен ответ, а не "ошибка":
    # заявка уже в таблице, сессию закроет вызывающий код по lead_saved.
    if result.lead_saved and not result.text:
        result.text = "Заявка принята. Спасибо, менеджер скоро свяжется с вами."
    return result
