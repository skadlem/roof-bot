"""Агент-цикл Gemini на LangGraph: диалог с инструментами (lookup_pricing, create_lead).

Первый ход сообщение клиента оборачивается контекстом RAG-базы (шаг 2:
rag.retrieve + rag.prompts). Дальше граф: модель с инструментами → выполнение
инструментов → результат обратно → следующий ход — пока модель не ответит
текстом или не исчерпает MAX_TURNS обращений к Gemini.

Граф: retrieve → model ⇄ tools (route по tool_calls и счётчику turns).
Используется ботом (main.py — модель через get_model(SYSTEM_PROMPT), свой
промпт SPIN-продаж) и CLI (rag/agent_cli.py — DEFAULT_SYSTEM_PROMPT).
Бот вызывает run_agent под per-phone lock (main.handle_single_message),
так что параллельных вызовов на одну сессию нет.
"""

import base64
import json
import os
import re
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, TypedDict

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

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


# Один экземпляр модели на процесс: у ChatGoogleGenerativeAI нет системного
# промпта в конструкторе, его хранит ChatSession (см. ниже). Ключ берётся из
# GOOGLE_API_KEY или GEMINI_API_KEY (как в .env бота). temperature=None —
# как в старом цикле: конфиг генерации не отправлялся, действует серверный
# дефолт (в LangChain дефолт 0.7 — это изменение поведения).
_model: ChatGoogleGenerativeAI | None = None


def get_model(system_prompt=None):
    """Модель агента; системный промпт (если есть) живёт в ChatSession."""
    global _model
    if _model is None:
        _model = ChatGoogleGenerativeAI(model=MODEL_NAME, temperature=None)
    return _model


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
    """Выполняет инструмент по имени; возвращает текст результата для модели."""
    if name == "lookup_pricing":
        return lookup_pricing(args.get("service"))
    if name == "create_lead":
        return create_lead(
            args.get("name"), args.get("phone"), args.get("service"),
            args.get("message"), default_phone=client_id, record_lead=record_lead,
        )
    return f"Ошибка: неизвестный инструмент «{name}»."


# Декларации инструментов для модели: pydantic-схемы аргументов + docstring =
# описание функции. Тексты те же, что были в легаси-декларациях TOOLS
# google-generativeai (0.8.x).
class _LookupPricingArgs(BaseModel):
    service: str = Field(
        description=(
            "Название услуги/материала (металлочерепица, "
            "профнастил, фальц, мягкая кровля и т.п.)"
        )
    )


class _CreateLeadArgs(BaseModel):
    name: str = Field(description="Имя клиента")
    phone: str = Field(
        default="",
        description=(
            "Телефон клиента в международном формате. "
            "Можно не указывать — подставится номер клиента автоматически"
        ),
    )
    service: str = Field(description="Услуга/материал, который интересует клиента")
    message: str = Field(description="Краткое описание запроса клиента")


def _make_tools(client_id, record_lead):
    """Инструменты одного прогона: create_lead замыкается на client_id и
    record_lead этого сообщения (всё как в _execute_calls старого цикла)."""

    @tool("lookup_pricing", args_schema=_LookupPricingArgs)
    def lookup_pricing_tool(service: str) -> str:
        """Цена услуги за кв.м из прайс-листа компании. Вызывай,
        когда клиент спрашивает цену или стоимость услуги."""
        return lookup_pricing(service)

    @tool("create_lead", args_schema=_CreateLeadArgs)
    def create_lead_tool(name: str, phone: str, service: str, message: str) -> str:
        """Сохранить лид клиента: имя, телефон, услуга, сообщение.
        Вызывай, когда клиент готов оставить заявку или подтвердил заказ.
        Телефон можно не указывать — подставится номер клиента автоматически."""
        return create_lead(
            name, phone, service, message,
            default_phone=client_id, record_lead=record_lead,
        )

    return [lookup_pricing_tool, create_lead_tool]


# --- СЕССИЯ И ГРАФ ---


@dataclass
class AgentResult:
    """Итог обработки одного сообщения клиента."""
    text: str = ""              # финальный ответ модели (пусто — ответа нет)
    lead_saved: bool = False    # create_lead выполнен успешно → сессию закрываем
    tool_calls: list = field(default_factory=list)  # (имя, args, результат) — для CLI
    hits: list = None           # чанки базы знаний первого хода — для CLI


def message_text(message) -> str:
    """Текст сообщения модели: строка или список text-блоков Gemini 3+."""
    if isinstance(message, str):
        return message
    content = getattr(message, "content", "")
    if isinstance(content, str):
        return content
    return "".join(
        block.get("text", "") for block in (content or [])
        if isinstance(block, dict) and block.get("type") == "text"
    )


def _from_disk_history(history):
    """Дисковая история [{"role": "user"|"model", "parts": [текст]}] → LangChain."""
    messages = []
    for item in history:
        text = "\n".join(str(p) for p in item.get("parts", []))
        if item.get("role") == "user":
            messages.append(HumanMessage(content=text))
        elif item.get("role") == "model":
            messages.append(AIMessage(content=text))
        else:
            raise ValueError(f"Неизвестная роль в истории: {item.get('role')!r}")
    return messages


class ChatSession:
    """Сессия диалога: модель + системный промпт + история (LangChain-сообщения).

    Заменяет genai ChatSession: история в chat.messages, системный промпт —
    отдельным полем (у ChatGoogleGenerativeAI его нет в конструкторе).
    send_message не трогает лимитер — это работа вызывающего кода, как было
    с chat.send_message в старом цикле и follow-up.
    """

    def __init__(self, model, system_prompt=None, history=None):
        self.model = model
        self.system_prompt = system_prompt
        self.messages = _from_disk_history(history) if history else []

    def send_message(self, message):
        """Один ход: сообщение (строка или HumanMessage) + ответ модели."""
        if isinstance(message, str):
            message = HumanMessage(content=message)
        msgs = list(self.messages)
        if self.system_prompt:
            msgs.insert(0, SystemMessage(content=self.system_prompt))
        msgs.append(message)
        response = self.model.invoke(msgs)
        self.messages.append(message)
        self.messages.append(response)
        return response


class _GraphState(TypedDict, total=False):
    session: ChatSession
    client_id: str
    user_message: str
    audio_data: bytes
    mime_type: str
    record_lead: Any
    kb_top_k: int
    tools: list
    result: AgentResult
    turns: int
    messages: list


def _retrieve_node(state):
    """Первый ход: RAG-контекст для текста, сырое аудио для голоса."""
    session = state["session"]
    if state.get("audio_data") is not None:
        b64 = base64.b64encode(state["audio_data"]).decode("ascii")
        first = HumanMessage(content=[
            {"type": "audio", "base64": b64, "mime_type": state.get("mime_type")}
        ])
        hits = None
    else:
        question = state["user_message"]
        try:
            hits = rag_retrieve.retrieve(
                question, state["client_id"], k=state.get("kb_top_k", KB_TOP_K)
            )
            first = HumanMessage(content=build_rag_message(question, hits))
        except Exception as e:
            print(f"[RAG RETRIEVE ERROR] {state['client_id']}: {e}")
            first, hits = HumanMessage(content=question), None
    state["result"].hits = hits
    return {"messages": list(session.messages) + [first]}


def _model_node(state):
    """Один ход модели: системный промпт + история → ответ с инструментами."""
    session = state["session"]
    msgs = list(state["messages"])
    if session.system_prompt:
        msgs.insert(0, SystemMessage(content=session.system_prompt))
    gemini_rate_limiter.acquire()
    response = session.model.bind_tools(state["tools"]).invoke(msgs)
    return {
        "messages": state["messages"] + [response],
        "turns": state.get("turns", 0) + 1,
    }


def _route_after_model(state):
    """Есть function_calls в ответе — выполняем инструменты, иначе всё."""
    last = state["messages"][-1]
    return "tools" if getattr(last, "tool_calls", None) else END


def _tools_node(state):
    """Выполняет function_calls модели; результаты — ToolMessage в историю.

    Дубль create_lead в одном цикле не выполняется (result.lead_saved уже
    True): заявка в таблице, повторный вызов задвоил бы лида. Об успехе
    судим по тексту инструмента: ошибки валидации начинаются с «Ошибка:»
    и не закрывают сессию.
    """
    result = state["result"]
    last = state["messages"][-1]
    messages = list(state["messages"])
    for call in last.tool_calls:
        name = call["name"]
        args = call.get("args") or {}
        if name == "create_lead" and result.lead_saved:
            outcome = (
                "Заявка уже сохранена ранее. Поблагодари клиента и скажи, "
                "что менеджер свяжется с ним."
            )
        else:
            outcome = _execute_tool(name, args, state["client_id"], state.get("record_lead"))
        if name == "create_lead" and not outcome.startswith("Ошибка:"):
            result.lead_saved = True
        result.tool_calls.append((name, args, outcome))
        # ToolMessage → FunctionResponse(name, response={"result": ...}) — та же
        # форма результата, что получала модель в старом цикле.
        messages.append(ToolMessage(
            content=json.dumps({"result": outcome}, ensure_ascii=False),
            tool_call_id=call["id"],
        ))
    return {"messages": messages}


def _route_after_tools(state):
    """Ответы инструментов обратно модели; на MAX_TURNS останавливаемся —
    невыполненные calls (create_lead в самом конце) уже выполнены здесь,
    без нового обращения к Gemini, чтобы заявка не потерялась."""
    return "model" if state.get("turns", 0) < MAX_TURNS else END


def _build_graph():
    g = StateGraph(_GraphState)
    g.add_node("retrieve", _retrieve_node)
    g.add_node("model", _model_node)
    g.add_node("tools", _tools_node)
    g.add_edge(START, "retrieve")
    g.add_edge("retrieve", "model")
    g.add_conditional_edges("model", _route_after_model)
    g.add_conditional_edges("tools", _route_after_tools)
    return g.compile()


_graph = _build_graph()


def run_agent(chat, client_id, *, user_message=None, audio_data=None,
              mime_type=None, record_lead=None, kb_top_k=KB_TOP_K):
    """Полный агент-цикл для одного сообщения клиента в сессии chat.

    Первый ход: текст оборачивается контекстом базы знаний (то, что делал
    build_message_with_kb в шаге 2), аудио идёт base64-блоком. Дальше до
    MAX_TURNS обращений к Gemini: ответ с function_call → выполнение
    инструмента → результат обратно → следующий ход.

    Вызывается под per-phone lock бота (main.handle_single_message), поэтому
    параллельных вызовов на одну сессию нет; лимитер Gemini общий на все потоки.
    """
    result = AgentResult()
    state = {
        "session": chat,
        "client_id": client_id,
        "user_message": user_message,
        "audio_data": audio_data,
        "mime_type": mime_type,
        "record_lead": record_lead,
        "kb_top_k": kb_top_k,
        "tools": _make_tools(client_id, record_lead),
        "result": result,
        "turns": 0,
    }
    final = _graph.invoke(state)
    chat.messages = final["messages"]
    last_ai = next((m for m in reversed(final["messages"]) if isinstance(m, AIMessage)), None)
    result.text = message_text(last_ai) if last_ai else ""
    # Лид сохранён, а текст не пришёл (модель позвала create_lead последним
    # ходом и цикл исчерпал MAX_TURNS) — клиенту нужен ответ, а не "ошибка":
    # заявка уже в таблице, сессию закроет вызывающий код по lead_saved.
    if result.lead_saved and not result.text:
        result.text = "Заявка принята. Спасибо, менеджер скоро свяжется с вами."
    return result
