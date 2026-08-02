"""Инструменты агента: lookup_pricing (прайс из prices.json) и create_lead (валидация).

Без сети и без Gemini: прайс подкладывается в tmp_path, запись лида — в список.
"""

from rag.agent import create_lead, lookup_pricing


def test_lookup_pricing_exact_and_partial(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "prices.json").write_text(
        '{"металлочерепица": 850, "фальц": 1200}', encoding="utf-8"
    )

    assert "850 тг за кв.м" in lookup_pricing("металлочерепица")
    # частичное вхождение: название услуги внутри фразы клиента
    assert "850 тг за кв.м" in lookup_pricing("сколько стоит металлочерепица с монтажом")
    assert "1200 тг за кв.м" in lookup_pricing("фальцевая кровля")

    assert "не найдена" in lookup_pricing("черепица глиняная")
    assert "не указана" in lookup_pricing("")


def test_create_lead_validation(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    written = []

    result = create_lead(
        "Иван", "77001234567", "Фальц", "Кровля 100 м²",
        record_lead=written.append,
    )
    assert "Лид сохранён" in result
    assert written[0]["phone"] == "77001234567"

    # имя пустое — ошибка, лид не пишется
    assert create_lead("", "77001234567", "Фальц", "", record_lead=written.append) \
        .startswith("Ошибка:")
    # телефон не похож на номер — ошибка
    assert create_lead("Иван", "12", "Фальц", "", record_lead=written.append) \
        .startswith("Ошибка:")

    assert len(written) == 1, "валидный лид записался ровно один раз"


def test_create_lead_phone_falls_back_to_client_id(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    written = []

    result = create_lead(
        "Иван", None, "Фальц", "", default_phone="77001234567",
        record_lead=written.append,
    )
    assert "Лид сохранён" in result
    assert written[0]["phone"] == "77001234567"
