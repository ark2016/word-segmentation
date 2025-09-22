#!/usr/bin/env python3
"""
Скрипт для удаления колонки text_no_spaces из submission.csv
"""
import pandas as pd

def main():
    try:
        # Читаем файл
        df = pd.read_csv('submission.csv')
        print(f"Исходный файл: {len(df)} записей, колонки: {list(df.columns)}")

        # Удаляем колонку text_no_spaces если она есть
        if 'text_no_spaces' in df.columns:
            df = df.drop('text_no_spaces', axis=1)
            print("Колонка 'text_no_spaces' удалена")

        # Сохраняем результат
        df.to_csv('submission.csv', index=False)
        print(f"Файл сохранен: {len(df)} записей, колонки: {list(df.columns)}")

        # Показываем первые несколько строк
        print("\nПервые строки:")
        print(df.head())

    except Exception as e:
        print(f"Ошибка: {e}")

if __name__ == "__main__":
    main()