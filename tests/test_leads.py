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


# --- create_order: заказ, уведомление владельца, экспорт в таблицу ---

ORDER_ARGS = {
    "name": "Иван",
    "material": "Металлочерепица",
    "color": "Коричневый",
    "area": "100",
    "price": "350000",
    "address": "Алматы, ул. Тестовая 1",
}


def _active_lead(main, gemini, phone="77001234567"):
    with main.state_lock:
        main.open_leads[phone] = {
            "last_seen": time.time(),
            "followup_sent": False,
            "history": [],
        }
        main.chat_sessions[phone] = gemini


def test_create_order_flow(main, outbox, sheets, gemini):
    phone = "77001234567"
    _active_lead(main, gemini, phone)
    gemini.responses.append(make_function_response("create_order", dict(ORDER_ARGS)))

    main.process_gemini_response(phone, user_message="Согласен, заказываю")

    # владельцу ушёл шаблон new_order_notification
    templates = [x for x in outbox if x[0] == "template"]
    assert len(templates) == 1
    assert templates[0][2] == "new_order_notification"

    # заказ ушёл в таблицу с именем и телефоном клиента
    assert sheets.rows, "строка заказа должна попасть в таблицу"
    assert sheets.rows[0][0] == "Иван"
    assert sheets.rows[0][1] == phone

    # клиенту ушёл финальный ответ
    assert any(x[0] == "text" and x[1] == phone for x in outbox)

    # лид закрыт
    assert phone not in main.open_leads
    assert phone not in main.chat_sessions


def test_create_order_handles_missing_fields(main, outbox, sheets, gemini):
    """Модель может не заполнить все поля — код не должен упасть."""
    phone = "77001234567"
    _active_lead(main, gemini, phone)
    gemini.responses.append(make_function_response("create_order", {"name": "Иван"}))

    main.process_gemini_response(phone, user_message="Заказываю")

    assert sheets.rows
    assert sheets.rows[0][0] == "Иван"
    assert sheets.rows[0][6] == "Самовывоз", "незаполненный адрес подставляется по умолчанию"


def test_sheets_failure_tolerated(main, outbox, gemini, monkeypatch):
    """Google Sheets упал — сообщение обрабатывается, бот не падает."""
    def boom():
        raise RuntimeError("google sheets is down")

    monkeypatch.setattr(main, "get_sheet", boom)

    phone = "77001234567"
    _active_lead(main, gemini, phone)
    gemini.responses.append(make_function_response("create_order", dict(ORDER_ARGS)))

    main.process_gemini_response(phone, user_message="Заказываю")

    # владелец уведомлён, клиенту ответ ушёл, лид закрыт — как в штатном сценарии
    assert any(x[0] == "template" for x in outbox)
    assert any(x[0] == "text" and x[1] == phone for x in outbox)
    assert phone not in main.open_leads


def test_end_chat_closes_lead_without_order(main, outbox, gemini):
    phone = "77001234567"
    _active_lead(main, gemini, phone)
    gemini.responses.append(make_function_response("end_chat", {}))

    main.process_gemini_response(phone, user_message="Спасибо, не нужно")

    assert phone not in main.open_leads
    assert phone not in main.chat_sessions
    assert not any(x[0] == "template" for x in outbox), "при отказе заказ владельцу не шлётся"
