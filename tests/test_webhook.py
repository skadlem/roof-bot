"""GET verify-рукопожатие и POST-вебхук: подпись HMAC, JSON, дубликаты."""

import hashlib
import hmac
import json
import time

from tests.helpers import make_text_message, webhook_payload

SECRET = "test-app-secret"


def _sign(raw_body: bytes) -> str:
    digest = hmac.new(SECRET.encode(), raw_body, hashlib.sha256).hexdigest()
    return f"sha256={digest}"


# --- GET /webhook: рукопожатие Meta ---


def test_verify_handshake_ok(client):
    r = client.get("/webhook", params={
        "hub.mode": "subscribe",
        "hub.challenge": "123456",
        "hub.verify_token": "test-verify-token",
    })
    assert r.status_code == 200
    assert r.json() == 123456


def test_verify_handshake_wrong_token(client):
    r = client.get("/webhook", params={
        "hub.mode": "subscribe",
        "hub.challenge": "123456",
        "hub.verify_token": "wrong-token",
    })
    assert r.status_code == 403


def test_verify_handshake_wrong_mode(client):
    r = client.get("/webhook", params={
        "hub.mode": "unsubscribe",
        "hub.challenge": "123456",
        "hub.verify_token": "test-verify-token",
    })
    assert r.status_code == 403


def test_verify_handshake_bad_challenge(client):
    r = client.get("/webhook", params={
        "hub.mode": "subscribe",
        "hub.challenge": "not-a-number",
        "hub.verify_token": "test-verify-token",
    })
    assert r.status_code == 400


# --- POST /webhook: подпись ---


def test_webhook_valid_signature(client, outbox, gemini):
    payload = webhook_payload([make_text_message("77001234567", "Здравствуйте")])
    raw = json.dumps(payload).encode()

    r = client.post("/webhook", content=raw, headers={"X-Hub-Signature-256": _sign(raw)})
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}

    # обработка идёт в фоновом потоке — даём ей завершиться
    time.sleep(1.0)
    assert len(outbox) == 1
    kind, phone, _ = outbox[0]
    assert kind == "text"
    assert phone == "77001234567"


def test_webhook_invalid_signature(client, outbox, gemini):
    payload = webhook_payload([make_text_message("77001234567", "Здравствуйте")])
    raw = json.dumps(payload).encode()

    r = client.post("/webhook", content=raw, headers={"X-Hub-Signature-256": "sha256=deadbeef"})
    assert r.status_code == 403

    time.sleep(1.0)
    assert outbox == []


def test_webhook_tampered_body(client, outbox, gemini):
    """Подпись валидна для одного тела, но пришло другое — отказ."""
    payload = webhook_payload([make_text_message("77001234567", "Здравствуйте")])
    raw = json.dumps(payload).encode()
    sig = _sign(raw)

    tampered = raw + b"x"
    r = client.post("/webhook", content=tampered, headers={"X-Hub-Signature-256": sig})
    assert r.status_code == 403

    time.sleep(1.0)
    assert outbox == []


def test_webhook_bad_signature_prefix(client, outbox, gemini):
    payload = webhook_payload([make_text_message("77001234567", "Здравствуйте")])
    raw = json.dumps(payload).encode()

    r = client.post("/webhook", content=raw, headers={"X-Hub-Signature-256": "deadbeef"})
    assert r.status_code == 403


def test_webhook_missing_signature(client, outbox, gemini):
    payload = webhook_payload([make_text_message("77001234567", "Здравствуйте")])
    raw = json.dumps(payload).encode()

    r = client.post("/webhook", content=raw)
    assert r.status_code == 403


def test_webhook_no_secret_falls_back_open(client, main, monkeypatch):
    """Если WA_APP_SECRET не настроен — бот работает без проверки подписи (поведение из main.py)."""
    monkeypatch.setattr(main, "WA_APP_SECRET", None)
    payload = webhook_payload([make_text_message("77001234567", "Здравствуйте")])
    raw = json.dumps(payload).encode()

    r = client.post("/webhook", content=raw, headers={"X-Hub-Signature-256": "sha256=whatever"})
    assert r.status_code == 200


def test_webhook_invalid_json(client):
    raw = b"{not json"

    r = client.post("/webhook", content=raw, headers={"X-Hub-Signature-256": _sign(raw)})
    assert r.status_code == 400


# --- POST /webhook: дедупликация ---


def test_duplicate_webhook_delivery_processed_once(client, outbox, gemini):
    """Meta может повторить доставку — сообщение должно обработаться ровно один раз."""
    payload = webhook_payload([
        make_text_message("77001234567", "Привет", mid="wamid.dup-delivery")
    ])
    raw = json.dumps(payload).encode()
    headers = {"X-Hub-Signature-256": _sign(raw)}

    r1 = client.post("/webhook", content=raw, headers=headers)
    assert r1.status_code == 200
    r2 = client.post("/webhook", content=raw, headers=headers)
    assert r2.status_code == 200

    time.sleep(1.0)
    assert len(outbox) == 1
