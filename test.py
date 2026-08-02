import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

print("Доступные модели для вашего API-ключа:")
for m in genai.list_models():
    # Исправленная проверка
    if 'generateContent' in m.supported_generation_methods:
        print(f"- {m.name}")