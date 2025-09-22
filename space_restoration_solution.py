#!/usr/bin/env python3
"""
Решение для восстановления пропущенных пробелов в тексте
Использует только LLM для обработки
"""
import pandas as pd
import requests
import os
import time
from typing import List
import json


class SpaceRestoration:
    def __init__(self, vllm_url: str = "http://localhost:8000"):
        self.vllm_url = vllm_url

    def query_llm(self, text: str) -> str:
        """Запрос к LLM через vLLM API для получения позиций пробелов"""
        try:
            prompt = f"""Задача: найти позиции где нужно вставить пробелы в тексте без пробелов.

Верни только список чисел - позиции символов, после которых нужно вставить пробел.
Позиции считаются с 0.

Примеры:
Текст: куплюайфон14про
Позиции: [5, 11, 13]

Текст: ищудомвПодмосковье
Позиции: [3, 6, 7]

Текст: сдаюквартирусмебельюитехникой
Позиции: [4, 12, 13, 21, 22]

Текст: {text}
Позиции:"""

            payload = {
                "model": "QVikhr-3-1.7B-Instruction-noreasoning.Q4_K_M",
                "messages": [
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.1,
                "max_tokens": 256,
                "stop": ["\n", "Текст:"]
            }

            response = requests.post(
                f"{self.vllm_url}/v1/chat/completions",
                json=payload,
                timeout=60
            )

            if response.status_code == 200:
                result = response.json()
                content = result['choices'][0]['message']['content'].strip()

                # Очищаем результат и ищем список
                lines = content.split('\n')
                for line in lines:
                    line = line.strip()
                    if line and ('[' in line and ']' in line):
                        return line

                return content
            else:
                print(f"LLM API error: {response.status_code}, {response.text}")
                return ""

        except Exception as e:
            print(f"Error querying LLM: {e}")
            return ""

    def parse_positions_from_llm_response(self, response: str) -> List[int]:
        """Парсит список позиций из ответа LLM"""
        try:
            import re

            # Ищем список в квадратных скобках
            match = re.search(r'\[([0-9,\s]*)\]', response)
            if not match:
                print(f"Не найден список в ответе: {response}")
                return []

            # Извлекаем числа
            numbers_str = match.group(1)
            if not numbers_str.strip():
                return []

            # Парсим числа
            positions = []
            for num_str in numbers_str.split(','):
                num_str = num_str.strip()
                if num_str:
                    try:
                        positions.append(int(num_str))
                    except ValueError:
                        continue

            return sorted(positions)

        except Exception as e:
            print(f"Ошибка парсинга позиций: {e}")
            return []

    def restore_spaces(self, text: str) -> List[int]:
        """
        Основной метод восстановления пробелов через LLM
        """
        if not text:
            return []

        # Получаем результат от LLM
        llm_result = self.query_llm(text)

        if not llm_result:
            print(f"LLM не вернул результат для: {text}")
            return []

        # Парсим позиции из ответа LLM
        positions = self.parse_positions_from_llm_response(llm_result)

        # Фильтруем позиции в пределах текста
        valid_positions = [pos for pos in positions if 0 < pos < len(text)]

        print(f"Текст: {text}")
        print(f"LLM результат: {llm_result}")
        print(f"Позиции: {valid_positions}")
        print("-" * 40)

        return valid_positions

    def calculate_f1_score(self, predicted: List[int], actual: List[int]) -> float:
        """Вычисляет F1-score для позиций пробелов"""
        if not actual and not predicted:
            return 1.0
        if not actual or not predicted:
            return 0.0

        predicted_set = set(predicted)
        actual_set = set(actual)

        intersection = len(predicted_set & actual_set)

        precision = intersection / len(predicted_set) if predicted_set else 0
        recall = intersection / len(actual_set) if actual_set else 0

        if precision + recall == 0:
            return 0.0

        f1 = 2 * (precision * recall) / (precision + recall)
        return f1


def main():
    """Основная функция обработки датасета"""

    # Инициализация модели
    vllm_url = os.getenv('VLLM_API_URL', 'http://localhost:8000')
    model = SpaceRestoration(vllm_url)

    # Проверяем доступность API
    try:
        test_response = requests.get(f"{vllm_url}/health", timeout=10)
        if test_response.status_code != 200:
            print(f"VLLM API недоступен по адресу {vllm_url}")
            print("Убедитесь, что vLLM сервер запущен")
            return
    except Exception as e:
        print(f"Не удается подключиться к VLLM API: {e}")
        print("Убедитесь, что vLLM сервер запущен")
        return

    # Загрузка датасета
    print("Загружаем датасет...")
    try:
        df = pd.read_csv('dataset_1937770_3.txt')
        print(f"Загружено {len(df)} записей")
    except Exception as e:
        print(f"Ошибка загрузки датасета: {e}")
        return

    # Обработка данных
    results = []
    processed = 0

    print("Начинаем обработку...")
    start_time = time.time()

    for idx, row in df.iterrows():
        text_no_spaces = row['text_no_spaces']

        # Восстанавливаем пробелы через LLM
        predicted_positions = model.restore_spaces(text_no_spaces)

        # Сохраняем результат
        results.append({
            'id': row['id'],
            'text_no_spaces': text_no_spaces,
            'predicted_positions': str(predicted_positions)
        })

        processed += 1
        if processed % 5 == 0:
            elapsed = time.time() - start_time
            avg_time = elapsed / processed
            remaining = (len(df) - processed) * avg_time
            print(f"Обработано: {processed}/{len(df)} ({processed/len(df)*100:.1f}%) "
                  f"Среднее время: {avg_time:.2f}с, Осталось: {remaining/60:.1f} мин")

        # Небольшая пауза чтобы не перегружать API
        time.sleep(0.1)

    # Сохранение результатов
    result_df = pd.DataFrame(results)
    result_df.to_csv('submission.csv', index=False)
    print(f"Результаты сохранены в submission.csv")

    print(f"Обработка завершена за {(time.time() - start_time)/60:.1f} минут")


if __name__ == "__main__":
    main()