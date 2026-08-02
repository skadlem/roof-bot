"""Прогон golden set через агент-цикл бота и оценку Gemini-судьёй.

Каждый кейс из evals/golden_set.jsonl:
  1. запускается напрямую через rag.agent.run_agent (без WhatsApp) с
     record_lead=None — create_lead валидирует заказ, но не пишет в реальную
     Google Sheets;
  2. ответ бота вместе с evidence (извлечённые фрагменты KB + результаты
     инструментов) оценивается судьёй по двум критериям:
     grounded — нет утверждений сверх evidence;
     correct — семантически совпадает с expected_answer;
  3. судья — та же модель/семейство SDK, что и бот, промпт судьи лежит в
     evals/judge_prompt.txt и правится без правки кода.

В конце печатается таблица проходимости (общая и по группам kb/tools) и
список проваленных кейсов. Выходной код 1, если grounded или correct ниже
70% — для CI. Выходной код 2, если часть кейсов не удалось оценить.

Запуск: venv/Scripts/python.exe -m evals.run_evals
"""

import json
import os
import re
import sys

from dotenv import load_dotenv

# как main.py: сначала configure, потом любые обращения к genai
load_dotenv()
import google.generativeai as genai  # noqa: E402
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

from rag import agent  # noqa: E402
from rag.prompts import build_bot_system_prompt  # noqa: E402

# Тот же системный промпт, что у бота (main.py): SPIN-продажи + прайс.
# main.py импортировать нельзя — на импорте он конструирует модель (сеть).
prices_text = "\n".join(
    f"- {k}: {v} тг за кв.м" for k, v in agent.load_prices().items())
BOT_SYSTEM_PROMPT = build_bot_system_prompt(prices_text)

EVALS_DIR = os.path.dirname(__file__)
GOLDEN_FILE = os.path.join(EVALS_DIR, "golden_set.jsonl")
JUDGE_PROMPT_FILE = os.path.join(EVALS_DIR, "judge_prompt.txt")

PASS_THRESHOLD = 70  # процентов; ниже — код возврата 1

JUDGE_MODEL_NAME = "models/gemini-3.1-flash-lite"


def load_golden_set():
    cases = []
    with open(GOLDEN_FILE, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                cases.append(json.loads(line))
    return cases


def run_case(case):
    """Агент-цикл на один кейс: без WhatsApp, без записи в таблицу."""
    chat = agent.get_model(system_prompt=BOT_SYSTEM_PROMPT).start_chat(history=[])
    result = agent.run_agent(
        chat, case["client_id"],
        user_message=case["question"],
        record_lead=None,  # лиды никуда не пишутся
    )
    evidence = {
        # прайс-лист — легитимный источник бота: он вшит в системный промпт,
        # lookup_pricing возвращает те же цифры (см. rag/prompts.py)
        "price_list": prices_text,
        "retrieved": [
            {"source": h["source"], "text": h["text"], "score": round(h["score"], 3)}
            for h in result.hits
        ],
        "tool_calls": [
            {"name": name, "args": plain_args(args), "result": res}
            for name, args, res in result.tool_calls
        ],
    }
    return result.text, evidence


def judge(question, expected_answer, bot_answer, evidence):
    """Оценка судьёй: JSON {grounded, correct, reason}."""
    with open(JUDGE_PROMPT_FILE, encoding="utf-8") as f:
        system_prompt = f.read()

    agent.gemini_rate_limiter.acquire()
    judge_model = genai.GenerativeModel(
        JUDGE_MODEL_NAME, system_instruction=system_prompt)
    payload = {
        "question": question,
        "expected_answer": expected_answer,
        "bot_answer": bot_answer,
        "evidence": evidence,
    }
    response = judge_model.generate_content(json.dumps(payload, ensure_ascii=False))
    return parse_judge_json(response.text)


def plain_args(args):
    """MapComposite из legacy SDK не сериализуется в JSON — привести к dict."""
    try:
        return dict(args)
    except (TypeError, ValueError):
        return str(args)


def parse_judge_json(text):
    """JSON из ответа судьи: пробуем целиком, затем вырезаем первый {...}."""
    for candidate in (text.strip(), extract_json(text)):
        if not candidate:
            continue
        try:
            data = json.loads(candidate)
            return data.get("grounded"), data.get("correct"), data.get("reason")
        except (json.JSONDecodeError, AttributeError):
            continue
    return None, None, text.strip()[:200]


def extract_json(text):
    m = re.search(r"\{.*\}", text, re.DOTALL)
    return m.group(0) if m else None


def pass_rate(passed, total):
    return 100.0 * passed / total if total else 0.0


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    cases = load_golden_set()
    print(f"Кейсов: {len(cases)}")

    verdicts = []
    for i, case in enumerate(cases, 1):
        print(f"[{i}/{len(cases)}] {case['grounded_in']} {case['question'][:60]!r}...")
        try:
            bot_answer, evidence = run_case(case)
        except Exception as e:
            verdicts.append({"case": case, "error": str(e)})
            print(f"  ОШИБКА прогона: {e}")
            continue

        grounded, correct, reason = judge(
            case["question"], case["expected_answer"], bot_answer, evidence)
        if grounded is None:
            verdicts.append({"case": case, "error": f"судья не вернул JSON: {reason}"})
            print(f"  ОШИБКА судьи: {reason}")
            continue

        verdicts.append({
            "case": case, "bot_answer": bot_answer, "evidence": evidence,
            "grounded": grounded == "да", "correct": correct == "да",
            "reason": reason,
        })
        print(f"  grounded={'+' if grounded=='да' else '-'} correct={'+' if correct=='да' else '-'} | {reason[:100]}")

    evaluated = [v for v in verdicts if "error" not in v]
    failed = [v for v in verdicts if "error" in v]
    total = len(evaluated)

    g_rate = pass_rate(sum(1 for v in evaluated if v["grounded"]), total)
    c_rate = pass_rate(sum(1 for v in evaluated if v["correct"]), total)

    print("\n" + "=" * 60)
    print(f"grounded: {g_rate:.0f}%  correct: {c_rate:.0f}%  (из {total} кейсов)")
    for group in ("kb", "tools"):
        g = [v for v in evaluated if v["case"]["grounded_in"] == group]
        gg = pass_rate(sum(1 for v in g if v["grounded"]), len(g))
        gc = pass_rate(sum(1 for v in g if v["correct"]), len(g))
        print(f"  {group:5}: grounded {gg:.0f}%  correct {gc:.0f}%  (n={len(g)})")

    if failed:
        print(f"\nНе оценено: {len(failed)}")
        for v in failed:
            print(f"  {v['case']['question'][:50]!r}: {v['error'][:120]}")

    if evaluated:
        print("\nПроваленные кейсы:")
        for v in evaluated:
            if not (v["grounded"] and v["correct"]):
                print(f"  - {v['case']['grounded_in']} {v['case']['question'][:50]!r}")
                print(f"    ожидалось: {v['case']['expected_answer'][:100]}")
                print(f"    ответ:     {v['bot_answer'][:100]}")
                print(f"    судья:     {v['reason'][:140]}")

    code = 0
    if g_rate < PASS_THRESHOLD or c_rate < PASS_THRESHOLD:
        code = 1
    elif failed:
        code = 2
    print(f"\nВыходной код: {code}")
    return code


if __name__ == "__main__":
    sys.exit(main())
