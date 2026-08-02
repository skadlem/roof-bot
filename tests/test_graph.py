"""Граф агента (rag.agent.run_agent) на скриптованных ответах FakeLLM.

Каждый тест прогоняет полный цикл: retrieve → model ⇄ tools — как бот
вызывает run_agent под per-phone lock.
"""

import json

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from rag import agent
from tests.helpers import FakeLLM, make_response, make_tool_call_response

LEAD_ARGS = {
    "name": "Иван",
    "phone": "77001234567",
    "service": "Металлочерепица",
    "message": "Кровля 100 м², хочу смету",
}


def _chat(llm):
    return agent.ChatSession(llm, system_prompt="тест")


def test_simple_reply_single_turn(main):
    llm = FakeLLM(responses=[make_response("Здравствуйте!")])
    result = agent.run_agent(_chat(llm), "77001234567", user_message="Здравствуйте")

    assert result.text == "Здравствуйте!"
    assert result.lead_saved is False
    assert result.tool_calls == []
    assert len(llm.invoked) == 1
    # системный промпт сессии идёт первым сообщением модели
    assert llm.invoked[0][0] == SystemMessage(content="тест")


def test_tools_bound_for_the_run(main):
    llm = FakeLLM()
    agent.run_agent(_chat(llm), "77001234567", user_message="Привет")
    assert [t.name for t in llm.bound_tools[0]] == ["lookup_pricing", "create_lead"]


def test_lookup_pricing_tool_loop(main, monkeypatch):
    monkeypatch.setattr(agent, "load_prices", lambda: {"металлочерепица": 3200})
    llm = FakeLLM(responses=[
        make_tool_call_response("lookup_pricing", {"service": "металлочерепица"}),
        make_response("Металлочерепица — 3200 тг за кв.м."),
    ])

    result = agent.run_agent(
        _chat(llm), "77001234567", user_message="Сколько стоит металлочерепица?"
    )

    assert result.text == "Металлочерепица — 3200 тг за кв.м."
    assert result.lead_saved is False
    assert result.tool_calls == [
        ("lookup_pricing", {"service": "металлочерепица"},
         "металлочерепица: 3200 тг за кв.м")
    ]
    # результат инструмента вернулся модели ToolMessage'ом в форме старого цикла
    tool_msg = next(m for m in llm.invoked[1] if isinstance(m, ToolMessage))
    assert json.loads(tool_msg.content) == {"result": "металлочерепица: 3200 тг за кв.м"}


def test_max_turns_stops_loop(main):
    llm = FakeLLM(responses=[
        make_tool_call_response("lookup_pricing", {"service": "металлочерепица"}),
        make_tool_call_response("lookup_pricing", {"service": "фальц"}),
        make_tool_call_response("lookup_pricing", {"service": "профнастил"}),
    ])

    result = agent.run_agent(_chat(llm), "77001234567", user_message="Цены?")

    assert len(llm.invoked) == agent.MAX_TURNS == 3, "больше MAX_TURNS к модели не ходим"
    assert len(result.tool_calls) == 3, "все вызовы инструментов выполнены"
    assert not result.lead_saved


def test_duplicate_create_lead_saved_once(main):
    saved = []

    def record(order):
        saved.append(order)
        return True

    llm = FakeLLM(responses=[
        make_tool_call_response("create_lead", dict(LEAD_ARGS)),
        make_tool_call_response("create_lead", dict(LEAD_ARGS)),
        make_response("Заявка уже сохранена."),
    ])

    result = agent.run_agent(
        _chat(llm), "77001234567", user_message="Согласен", record_lead=record
    )

    assert result.lead_saved is True
    assert len(saved) == 1, "create_lead выполняется ровно один раз"
    assert result.tool_calls[1][2].startswith("Заявка уже сохранена ранее")
    assert result.text == "Заявка уже сохранена."


def test_lead_saved_without_final_text_gets_fallback(main):
    saved = []

    def record(order):
        saved.append(order)
        return True

    # последний ход модели — create_lead: цикл исчерпал MAX_TURNS, текст
    # не пришёл, но заявка в таблице — клиенту уходит ответ-заглушка
    llm = FakeLLM(responses=[
        make_tool_call_response("lookup_pricing", {"service": "фальц"}),
        make_tool_call_response("create_lead", dict(LEAD_ARGS)),
        make_tool_call_response("create_lead", dict(LEAD_ARGS)),
    ])

    result = agent.run_agent(
        _chat(llm), "77001234567", user_message="Заказываю", record_lead=record
    )

    assert result.lead_saved is True
    assert len(saved) == 1
    assert len(llm.invoked) == agent.MAX_TURNS
    assert result.text == "Заявка принята. Спасибо, менеджер скоро свяжется с вами."


def test_rate_limiter_acquired_per_model_call(main, monkeypatch):
    class Counter:
        def __init__(self):
            self.calls = 0

        def acquire(self):
            self.calls += 1

    limiter = Counter()
    monkeypatch.setattr(agent, "gemini_rate_limiter", limiter)
    llm = FakeLLM(responses=[
        make_tool_call_response("lookup_pricing", {"service": "металлочерепица"}),
        make_response("Цена: 3200 тг за кв.м"),
    ])
    monkeypatch.setattr(agent, "load_prices", lambda: {"металлочерепица": 3200})

    result = agent.run_agent(_chat(llm), "77001234567", user_message="Сколько?")

    assert limiter.calls == 2, "лимитер берётся на каждое обращение к модели"
    assert result.text == "Цена: 3200 тг за кв.м"


def test_rag_error_falls_back_to_plain_question(main, monkeypatch):
    def boom(question, client_id, k=4):
        raise RuntimeError("no api")

    monkeypatch.setattr(agent.rag_retrieve, "retrieve", boom)
    llm = FakeLLM()

    result = agent.run_agent(_chat(llm), "77001234567", user_message="гарантия?")

    assert result.hits is None
    assert llm.invoked[0][-1].content == "гарантия?"


def test_audio_goes_as_base64_block_without_rag(main, monkeypatch):
    calls = []

    def boom(question, client_id, k=4):
        calls.append(question)
        raise RuntimeError("no api")

    monkeypatch.setattr(agent.rag_retrieve, "retrieve", boom)
    llm = FakeLLM()

    result = agent.run_agent(
        _chat(llm), "77001234567", audio_data=b"audio", mime_type="audio/ogg"
    )

    assert result.hits is None
    assert calls == [], "голос в базу знаний не ходит"
    last = llm.invoked[0][-1]
    assert isinstance(last, HumanMessage)
    assert last.content == [{"type": "audio", "base64": "YXVkaW8=", "mime_type": "audio/ogg"}]


def test_session_history_accumulates(main):
    llm = FakeLLM(responses=[make_response("Первый ответ"), make_response("Второй ответ")])
    chat = _chat(llm)

    agent.run_agent(chat, "77001234567", user_message="Привет")
    agent.run_agent(chat, "77001234567", user_message="Как дела?")

    assert len(llm.invoked) == 2
    # второй ход видит всю историю: оба вопроса и первый ответ
    sent = llm.invoked[1]
    assert any(isinstance(m, HumanMessage) and m.content == "Привет" for m in sent)
    assert any(isinstance(m, HumanMessage) and m.content == "Как дела?" for m in sent)
    assert any(isinstance(m, AIMessage) and m.content == "Первый ответ" for m in sent)
    assert chat.messages[-1].content == "Второй ответ"
