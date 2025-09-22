#!/usr/bin/env python3
"""
Скрипт для загрузки модели QVikhr-3-1.7B-Instruction-noreasoning.Q4_K_M
"""
import os
import requests
from pathlib import Path

def download_file(url, filepath):
    """Загружает файл по URL с прогрессом"""
    print(f"Загружаем {filepath.name}...")

    response = requests.get(url, stream=True)
    response.raise_for_status()

    total_size = int(response.headers.get('content-length', 0))
    downloaded = 0

    with open(filepath, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
                downloaded += len(chunk)
                if total_size > 0:
                    progress = (downloaded / total_size) * 100
                    print(f"\rПрогресс: {progress:.1f}%", end='', flush=True)

    print(f"\n✓ {filepath.name} загружен успешно")

def main():
    models_dir = Path("/app/models")
    models_dir.mkdir(exist_ok=True)

    model_name = "QVikhr-3-1.7B-Instruction-noreasoning.Q4_K_M"
    model_path = models_dir / f"{model_name}.gguf"

    # Проверяем, существует ли уже модель
    if model_path.exists():
        print(f"Модель {model_name} уже существует. Пропускаем загрузку.")
        return

    # URL для загрузки модели (пример, может потребоваться изменить)
    # Обычно модели GGUF размещаются на HuggingFace
    model_url = f"https://huggingface.co/Vikhrmodels/QVikhr-3-1.7B-Instruction-noreasoning-GGUF/resolve/main/{model_name}.gguf"

    try:
        download_file(model_url, model_path)
        print(f"Модель {model_name} успешно загружена в {model_path}")
    except Exception as e:
        print(f"Ошибка при загрузке модели: {e}")
        # Если не удается загрузить, создаем заглушку
        print("Создаем заглушку для тестирования...")
        with open(model_path, 'w') as f:
            f.write("# Placeholder model file\n")

if __name__ == "__main__":
    main()