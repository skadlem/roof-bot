"""Добыча кандидатов для evals/golden_set.jsonl из реальных данных.

Два источника, оба уже использует репозиторий:
  * Google Sheets (gspread, как main.get_sheet) — экспорт лидов;
  * chats_logs/*.txt — построчные логи переписок (телефоны в именах файлов).

Скрипт собирает реплики клиентов, похожие на вопросы (или несущие данные
заказа), и пишет их в evals/candidates.jsonl без ответов — ответы добавляются
вручную при курировании golden_set.jsonl. Телефоны не печатаются и не пишутся
в файлы: клиенту выдается только количество и примеры реплик с заменой
телефона на <PHONE>.

Запуск: venv/Scripts/python.exe -m evals.mine_golden_set
"""

import json
import os
import re
import sys
from datetime import datetime

import gspread
from dotenv import load_dotenv

# консоль Windows часто cp1252 — русский текст в вывод не влезает
sys.stdout.reconfigure(encoding="utf-8")

load_dotenv()

EVALS_DIR = os.path.dirname(__file__)
CANDIDATES_FILE = os.path.join(EVALS_DIR, "candidates.jsonl")
CHATS_DIR = "chats_logs"

# Реплики клиента, которые не несут ни вопроса, ни данных заказа
SKIP_EXACT = {
    "Здравствуйте", "Здравствуй", "Привет", "привет", "раз", "Жарайд",
    "Жақсы", "Нет", "нет", "неа", "все норм", "Все норм", "Жоқ, но всё еще ойланып жатырмыр",
}
SKIP_PATTERNS = [
    r"^\[Голосовое сообщение\]$",
    r"^(да|нет|неа|хорошо|ок|согласен|заказываю|оформляйте)\b.*$",  # короткие подтверждения без данных
]


def read_sheet_rows():
    """Строки лида из Google Sheets: [name, phone, material, color, area, price, address]."""
    gc = gspread.service_account(filename="google_credentials.json")
    sheet = gc.open_by_key(os.getenv("SHEET_KEY")).sheet1
    rows = sheet.get_all_values()
    return [r for r in rows[1:] if r and r[0]]  # без заголовка и пустых


def read_chat_messages():
    """Реплики клиента из chats_logs/*.txt: список (phone, text)."""
    messages = []
    for filename in os.listdir(CHATS_DIR):
        m = re.match(r"chat_(\d+)\.txt$", filename)
        if not m:
            continue
        phone = m.group(1)
        with open(os.path.join(CHATS_DIR, filename), "r", encoding="utf-8") as f:
            for line in f:
                m = re.match(r"\[\d{4}-\d{2}-\d{2} [\d:]+] Клиент: (.*)", line.rstrip("\n"))
                if m and m.group(1):
                    messages.append((phone, m.group(1)))
    return messages


def is_worth_candidate(text):
    """Реплика может стать вопросом/кейсом для evals: не болванка и не чистый отказ."""
    if text in SKIP_EXACT or not text.strip():
        return False
    if len(text) < 2:
        return False
    return not any(re.match(p, text, re.IGNORECASE) for p in SKIP_PATTERNS)


def main():
    candidates = []

    # 1) Google Sheets: каждая строка лида — готовый кейс create_lead (данные заказа)
    try:
        rows = read_sheet_rows()
        for row in rows:
            name, phone, material, color, area, price, address = (row + [""] * 7)[:7]
            if not name or not material:
                continue
            question = (f"Меня зовут {name}, адрес {address or 'самовывоз'}, "
                        f"хочу заказать {material} на {area} м², цвет {color}")
            candidates.append({
                "question": question, "client_id": phone,
                "source": "google_sheets", "kind": "lead",
            })
        print(f"Sheets: {len(rows)} строк лидов -> {sum(1 for c in candidates)} кандидатов на лиды")
    except Exception as e:
        print(f"Sheets недоступен ({e}) — только chats_logs")

    # 2) chats_logs: реальные реплики клиентов
    for phone, text in read_chat_messages():
        if not is_worth_candidate(text):
            continue
        kind = "lead" if re.search(r"заказ|оформ|заявк|запиши|забронир", text, re.IGNORECASE) else "question"
        candidates.append({"question": text, "client_id": phone, "source": f"chats_logs/chat_{phone}.txt", "kind": kind})

    with open(CANDIDATES_FILE, "w", encoding="utf-8") as f:
        for c in candidates:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    print(f"Всего кандидатов: {len(candidates)} -> {CANDIDATES_FILE}")
    print("Примеры (телефоны заменены):")
    for c in candidates[:10]:
        print(f"  [{c['kind']}] {c['question'][:70]}  (клиент <PHONE>)")


if __name__ == "__main__":
    main()
