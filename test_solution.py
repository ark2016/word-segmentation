#!/usr/bin/env python3
"""
Тестирование LLM-only решения
"""
import requests
import os
from space_restoration_solution import SpaceRestoration


def test_api_connection():
    """Проверяет подключение к vLLM API"""
    vllm_url = os.getenv('VLLM_API_URL', 'http://localhost:8000')

    try:
        response = requests.get(f"{vllm_url}/health", timeout=10)
        if response.status_code == 200:
            print("✓ vLLM API доступен")
            return True
        else:
            print(f"✗ vLLM API вернул статус {response.status_code}")
            return False
    except Exception as e:
        print(f"✗ Не удается подключиться к vLLM API: {e}")
        return False


def test_llm_examples():
    """Тестирует LLM на примерах из ТЗ"""

    vllm_url = os.getenv('VLLM_API_URL', 'http://localhost:8000')
    model = SpaceRestoration(vllm_url)

    test_cases = [
        "куплюайфон14про",
        "ищудомвПодмосковье",
        "сдаюквартирусмебельюитехникой",
        "новыйдивандоставканедорого",
        "куплютелевизорPhilips",
        "ищуработупрограммистом",
        "срочноотдамкотенка",
        "новаякуртказима",
    ]

    print("Тестирование LLM на примерах:")
    print("=" * 50)

    for text in test_cases:
        print(f"\nОбрабатываем: {text}")

        # Получаем результат от LLM
        llm_result = model.query_llm(text)
        print(f"LLM ответ: {llm_result}")

        # Парсим позиции
        positions = model.parse_positions_from_llm_response(llm_result)
        valid_positions = [pos for pos in positions if 0 < pos < len(text)]
        print(f"Позиции: {valid_positions}")

        print("-" * 30)


if __name__ == "__main__":
    print("Проверка LLM-only решения")
    print("=" * 40)

    if test_api_connection():
        test_llm_examples()
    else:
        print("\nДля тестирования нужно запустить vLLM сервер:")
        print("1. Скачайте модель: python download_model.py")
        print("2. Запустите сервер: docker-compose up vllm-server")
        print("3. Или локально: python -m vllm.entrypoints.openai.api_server --model <model_path>")