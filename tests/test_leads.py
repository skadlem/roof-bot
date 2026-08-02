"""Логика лидов: создание, обновление, follow-up, экспорт заказа в таблицу."""

import time

from tests.helpers import make_function_response


def test_new_phone_creates_lead(main, outbox, gemini):
    main.process_gemini_response("77001234567", user_message="Здравствуйте")

    lead = main.open_leads["77001234567"]
    assert lead["followup_sent"] is False
    assert time.time() - lead["last_seen"] < 5
    assert lead["history"], "история диалога должна сохраняться на диск"

    assert outbox[0][1] == "77001234567"


def test_existing_lead_refreshed(main, outbox, gemini):
    phone = "77001234567"
    with main.state_lock:
        main.open_leads[phone] = {
            "last_seen": time.time() - 5 * 3600,
            "followup_sent": True,
            "history": [],
        }

    main.process_gemini_response(phone, user_message="Я тут")

    lead = main.open_leads[phone]
    assert lead["followup_sent"] is False, "новое сообщение сбрасывает follow-up"
    assert time.time() - lead["last_seen"] < 5, "last_seen должен обновляться"


def test_history_restored_from_disk(main, outbox, monkeypatch):
    """После 'рестарта' сессия восстанавливается из сохранённой истории."""
    phone = "77001234567"
    saved = [{"role": "user", "parts": ["Здравствуйте"]}]
    with main.state_lock:
        main.open_leads[phone] = {
            "last_seen": time.time(),
            "followup_sent": False,
            "history": saved,
        }

    started_with = {}
    real_build = main.build_chat_session

    def spy_build(p):
        chat = real_build(p)
        started_with[p] = list(chat.history)
        return chat

    monkeypatch.setattr(main, "build_chat_session", spy_build)
    main.process_gemini_response(phone, user_message="Привет")

    assert started_with[phone] == saved, "сессия должна строиться на истории с диска"


# --- Follow-up ---

STALE = 4 * 3600 + 60          # 4 часа + запас
MAX_AGE = 48 * 3600 + 60       # 48 часов + запас


def test_followup_sent_for_stale_lead(main, outbox, gemini):
    phone = "77001234567"
    with main.state_lock:
        main.open_leads[phone] = {
            "last_seen": time.time() - STALE,
            "followup_sent": False,
            "history": [],
        }
        main.chat_sessions[phone] = gemini

    main.check_stale_leads()

    assert outbox, "follow-up должен уйти клиенту"
    assert outbox[0][0] == "text"
    assert main.open_leads[phone]["followup_sent"] is True


def test_no_followup_for_fresh_lead(main, outbox, gemini):
    phone = "77001234567"
    with main.state_lock:
        main.open_leads[phone] = {
            "last_seen": time.time(),
            "followup_sent": False,
            "history": [],
        }

    main.check_stale_leads()

    assert outbox == []


def test_no_second_followup(main, outbox, gemini):
    phone = "77001234567"
    with main.state_lock:
        main.open_leads[phone] = {
            "last_seen": time.time() - STALE,
            "followup_sent": True,
            "history": [],
        }

    main.check_stale_leads()

    assert outbox == [], "follow-up отправляется ровно один раз"


def test_very_old_lead_removed(main, outbox, gemini):
    phone = "77001234567"
    with main.state_lock:
        main.open_leads[phone] = {
            "last_seen": time.time() - MAX_AGE,
            "followup_sent": False,
            "history": [],
        }
        main.chat_sessions[phone] = gemini

    main.check_stale_leads()

    assert phone not in main.open_leads
    assert phone not in main.chat_sessions
    assert outbox == []


# --- create_lead: лид в таблицу, закрытие сессии (уведомления владельца больше нет) ---

LEAD_ARGS = {
    "name": "Иван",
    "phone": "77001234567",
    "service": "Металлочерепица",
    "message": "Кровля 100 м², хочу смету",
}


def _active_lead(main, gemini, phone="77001234567"):
    with main.state_lock:
        main.open_leads[phone] = {
            "last_seen": time.time(),
            "followup_sent": False,
            "history": [],
        }
        main.chat_sessions[phone] = gemini


def _lead_result(gemini):
    """Результат create_lead, который агент-цикл вернул модели."""
    content = gemini.sent_messages[1][0][0]
    return content.parts[0].function_response.response["result"]


def test_create_lead_flow(main, outbox, sheets, gemini):
    phone = "77001234567"
    _active_lead(main, gemini, phone)
    gemini.responses.append(make_function_response("create_lead", dict(LEAD_ARGS)))

    main.process_gemini_response(phone, user_message="Согласен, оставьте заявку")

    # лид ушёл в таблицу, владельцу — шаблон new_order_notification
    assert sheets.rows, "строка лида должна попасть в таблицу"
    assert sheets.rows[0][0] == "Иван"
    assert sheets.rows[0][1] == phone
    assert sheets.rows[0][2] == "Металлочерепица"
    templates = [x for x in outbox if x[0] == "template"]
    assert len(templates) == 1
    assert templates[0][1] == "79990000000"
    assert templates[0][2] == "new_order_notification"

    # клиенту ушёл финальный ответ
    assert any(x[0] == "text" and x[1] == phone for x in outbox)

    # лид закрыт
    assert phone not in main.open_leads
    assert phone not in main.chat_sessions


def test_create_lead_rejects_bad_phone(main, outbox, sheets, gemini):
    """Телефон не похож на номер — лид не пишется, ошибка уходит модели."""
    phone = "77001234567"
    _active_lead(main, gemini, phone)
    args = dict(LEAD_ARGS, phone="abc")
    gemini.responses.append(make_function_response("create_lead", args))

    main.process_gemini_response(phone, user_message="Запишите меня")

    assert sheets.rows == [], "с невалидным телефоном лид в таблицу не пишется"
    assert _lead_result(gemini).startswith("Ошибка:")
    assert any(x[0] == "text" and x[1] == phone for x in outbox)
    assert not any(x[0] == "template" for x in outbox), "с ошибкой валидации владельцу не шлём"
    # ошибка валидации сессию не закрывает — клиент остаётся в диалоге
    assert phone in main.open_leads
    assert phone in main.chat_sessions


def test_create_lead_missing_phone_falls_back_to_client(main, outbox, sheets, gemini):
    """Модель не передала телефон — подставляется номер клиента из сообщения."""
    phone = "77001234567"
    _active_lead(main, gemini, phone)
    gemini.responses.append(make_function_response(
        "create_lead", {"name": "Иван", "service": "Фальц", "message": "Хочу смету"}
    ))

    main.process_gemini_response(phone, user_message="Заказываю")

    assert sheets.rows[0][1] == phone, "телефон должен подставиться из client_id"
    assert phone not in main.open_leads


def test_sheets_failure_tolerated(main, outbox, gemini, monkeypatch):
    """Google Sheets упал — бот не падает: владелец не уведомляется, лид
    остаётся открытым, клиенту уходит сообщение об ошибке сохранения."""
    def boom():
        raise RuntimeError("google sheets is down")

    monkeypatch.setattr(main, "get_sheet", boom)

    phone = "77001234567"
    _active_lead(main, gemini, phone)
    gemini.responses.append(make_function_response("create_lead", dict(LEAD_ARGS)))

    main.process_gemini_response(phone, user_message="Заказываю")

    # владелец НЕ уведомлён (лида в таблице нет — уведомлять не о чем)
    assert not any(x[0] == "template" for x in outbox)
    # лид НЕ закрыт: таблица недоступна, сессия остаётся — клиент повторит позже
    assert phone in main.open_leads
    assert phone in main.chat_sessions
    # модель получила ошибку сохранения и передала её клиенту
    assert _lead_result(gemini).startswith("Ошибка:")
    assert any(x[0] == "text" and x[1] == phone for x in outbox)
