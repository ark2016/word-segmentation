# Используем официальный образ Python с поддержкой CUDA
FROM nvidia/cuda:11.8-devel-ubuntu22.04

# Устанавливаем Python и необходимые зависимости
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Обновляем pip
RUN python3 -m pip install --upgrade pip

# Устанавливаем PyTorch и необходимые библиотеки
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Устанавливаем vLLM и другие зависимости
RUN pip install vllm
RUN pip install transformers accelerate

# Устанавливаем дополнительные библиотеки для обработки данных
RUN pip install pandas numpy scikit-learn requests

# Создаем рабочую директорию
WORKDIR /app

# Копируем файлы проекта
COPY . /app/

# Создаем директорию для моделей
RUN mkdir -p /app/models

# Экспонируем порт для API
EXPOSE 8000

# Команда по умолчанию
CMD ["python3", "main.py"]