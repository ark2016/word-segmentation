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
    def __init__(self, ollama_url: str = "http://localhost:11434"):
        self.ollama_url = ollama_url
        self.model_name = "hf.co/Vikhrmodels/QVikhr-3-1.7B-Instruction-noreasoning-GGUF:Q4_K_M"

    def query_llm(self, text: str) -> str:
        """Запрос к LLM через Ollama API для получения позиций пробелов"""
        try:
            prompt = f"""<task>Задача: найти позиции где нужно вставить пробелы в тексте без пробелов (задача word segmentation). 
            Тебуется вернуть только ответ, а именно числа, перечисленные через запятую в квадратных скобках. </task>

            <note> ВАЖНО: Отвечай только списком чисел в квадратных скобках. Никаких объяснений, рассуждений или дополнительного текста.</note>
            <examples>
                <text>
                куплюайфон17max
                </text>
                <answer>
                [5, 11, 13]
                </answer>
                <text>
                ищудомвПодмосковье
                </text>
                <answer>
                [3, 6, 7]
                </answer>
                <text>
                сдаюквартирусмебельюитехникой
                </text>
                <answer>
                [4, 12, 13, 21, 22]
                </answer>
            </examples>
            <input>
                <text>
                {text}
                </text>
            </input>
            """

            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.0,
                    "top_p": 0.1,
                    "num_predict": 200,
                    "repeat_penalty": 1.0
                }
            }

            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json=payload,
                timeout=60
            )

            if response.status_code == 200:
                
                result = response.json()
                content = result['response'].strip()

                # Ищем список чисел в квадратных скобках с помощью регулярки
                import re
                content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()
                match = re.search(r'\[(\d+(?:,\s*\d+)*|)\]', content)
                if match:
                    return match.group(0)
                return content
            else:
                print(f"Ollama API error: {response.status_code}, {response.text}")
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
    ollama_url = os.getenv('OLLAMA_API_URL', 'http://localhost:11434')
    model = SpaceRestoration(ollama_url)

    # Проверяем доступность API
    try:
        test_response = requests.get(f"{ollama_url}/api/tags", timeout=10)
        if test_response.status_code != 200:
            print(f"Ollama API недоступен по адресу {ollama_url}")
            print("Убедитесь, что Ollama сервер запущен")
            return
    except Exception as e:
        print(f"Не удается подключиться к Ollama API: {e}")
        print("Убедитесь, что Ollama сервер запущен")
        return

    # Загрузка датасета
    print("Загружаем датасет...")
    try:
        df = pd.read_csv('dataset_1937770_3.txt', sep=',', quoting=3, on_bad_lines='skip')
        print(f"Загружено {len(df)} записей")
        print(f"Колонки: {list(df.columns)}")
        print(f"Первые несколько строк:")
        print(df.head())
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