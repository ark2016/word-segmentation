# Word Segmentation Solution

Решение для восстановления пропущенных пробелов в русском тексте с использованием LLM через Ollama.

## Выбор модели

Используется **QVikhr-3-1.7B-Instruction-noreasoning** - очень лёгкая модель (1.7B параметров), которая является SOTA среди русскоязычных моделей в своём классе. Оптимальное соотношение качества и скорости для задач NLP.

## Быстрый старт

### 1. Установка Ollama
```bash
curl -fsSL https://ollama.com/install.sh | sh
```

### 2. Запуск модели
```bash
ollama run hf.co/Vikhrmodels/QVikhr-3-1.7B-Instruction-noreasoning-GGUF:Q4_K_M
```

### 3. Подготовка окружения
```bash
python3 -m venv venv
source venv/bin/activate
pip install pandas numpy requests
```

### 4. Запуск обработки
```bash
export OLLAMA_API_URL=http://localhost:11434
python space_restoration_solution.py
```

## Результат

Файл `submission.csv` будет содержать:
- `id` - идентификатор записи
- `predicted_positions` - список позиций для вставки пробелов

**Для отправки**: переименуйте `submission_fixed_with_quotes.csv` в `.txt` формат для загрузки в систему.

## Требования

- GPU с 4GB+ VRAM
- CUDA совместимая видеокарта
- Python 3.8+

## Тестирование

```bash
python test_solution.py
```

## Подход к решению

- **LLM-only решение**: Используется только языковая модель без дополнительных алгоритмов
- **Zero-shot промптинг**: Модель работает на примерах из промпта без дообучения
- **Оптимизация промпта**: Специально настроенный промпт для минимизации рассуждений и получения точных позиций

## Производительность

- **Скорость**: ~2-5 текстов/сек на Tesla T4
- **Память**: ~4GB VRAM
- **Точность**: Хорошо работает на коротких текстах объявлений

## Устранение проблем

```bash
# Проверка работы Ollama
ollama ps

# Перезагрузка модели
ollama pull hf.co/Vikhrmodels/QVikhr-3-1.7B-Instruction-noreasoning-GGUF:Q4_K_M
```

## Структура проекта

```
word-segmentation/
├── space_restoration_solution.py  # Основное решение
├── test_solution.py              # Тесты и проверки
├── fix_submission.py             # Утилита для форматирования
├── requirements.txt              # Зависимости
├── dataset_1937770_3.txt         # Входные данные
└── README.md                     # Документация
```