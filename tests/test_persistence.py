"""JSON-персистентность: лиды и seen-сообщения, старый формат, битые файлы."""

import json
import os
import time


def test_leads_round_trip(main):
    phone = "77001234567"
    with main.state_lock:
        main.open_leads[phone] = {
            "last_seen": 123.0,
            "followup_sent": True,
            "history": [{"role": "user", "parts": ["Привет"]}],
        }
        main.save_leads(main.open_leads)

    reloaded = main.load_leads()
    assert reloaded[phone] == {
        "last_seen": 123.0,
        "followup_sent": True,
        "history": [{"role": "user", "parts": ["Привет"]}],
    }


def test_leads_empty_file(main):
    with open(main.LEADS_FILE, "w", encoding="utf-8") as f:
        f.write("")

    assert main.load_leads() == {}


def test_leads_legacy_format_normalized(main):
    """Старый формат {"phone": timestamp} приводится к новому."""
    with open(main.LEADS_FILE, "w", encoding="utf-8") as f:
        json.dump({"77001234567": 12345.5, "+77761112233": 999}, f)

    data = main.load_leads()
    assert data["77001234567"] == {
        "last_seen": 12345.5,
        "followup_sent": False,
        "history": [],
    }
    assert data["+77761112233"]["last_seen"] == 999.0


def test_leads_corrupted_file(main):
    with open(main.LEADS_FILE, "w", encoding="utf-8") as f:
        f.write("{broken json!!")

    assert main.load_leads() == {}


def test_leads_missing_file(main):
    if os.path.exists(main.LEADS_FILE):
        os.remove(main.LEADS_FILE)

    assert main.load_leads() == {}


def test_seen_messages_round_trip(main):
    main.seen_messages["wamid.x"] = time.time()
    main.save_seen_messages(main.seen_messages)

    loaded = main.load_seen_messages()
    assert "wamid.x" in loaded


def test_seen_messages_expired_dropped_on_load(main):
    old_ts = time.time() - main.SEEN_MESSAGE_TTL - 10
    with open(main.SEEN_MESSAGES_FILE, "w", encoding="utf-8") as f:
        json.dump({"wamid.old": old_ts}, f)

    assert main.load_seen_messages() == {}


def test_seen_messages_corrupted_file(main):
    with open(main.SEEN_MESSAGES_FILE, "w", encoding="utf-8") as f:
        f.write("nope")

    assert main.load_seen_messages() == {}
