"""Серийность сообщений одного номера и параллельность разных номеров."""

import threading

from tests.helpers import TrackingFakeLLM, Tracker, make_text_message


def _run_in_threads(main, monkeypatch, phones):
    tracker = Tracker()
    monkeypatch.setattr(main, "model", TrackingFakeLLM(tracker))
    errors = []

    def work(phone):
        try:
            main.handle_single_message(
                make_text_message(phone, "Привет", mid=f"wamid.{phone}")
            )
        except Exception as e:  # pragma: no cover — только для диагностики
            errors.append(e)

    threads = [threading.Thread(target=work, args=(p,)) for p in phones]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=15)

    assert not errors, f"обработка упала: {errors}"
    return tracker


def test_same_phone_messages_processed_serially(main, monkeypatch):
    """Два сообщения одного клиента не обрабатываются одновременно
    (per-phone lock защищает один Gemini ChatSession от гонок)."""
    tracker = _run_in_threads(main, monkeypatch, ["77001234567", "77001234567"])

    assert tracker.max_active == 1


def test_different_phones_processed_in_parallel(main, monkeypatch, outbox):
    """Разные клиенты обрабатываются параллельно, а не по очереди."""
    tracker = _run_in_threads(main, monkeypatch, ["77001234567", "77001112233"])

    assert tracker.max_active == 2
