import pandas as pd
import ast

# Читаем датасет чтобы понять какие ID должны быть
try:
    df_dataset = pd.read_csv('dataset_1937770_3.txt')
    print("=== АНАЛИЗ ДАТАСЕТА ===")
    print(f"Dataset содержит {len(df_dataset)} строк")
    print(f"ID в dataset: от {df_dataset['id'].min()} до {df_dataset['id'].max()}")
    
    # Получаем все необходимые ID
    required_ids = sorted(df_dataset['id'].unique())
    print(f"Всего уникальных ID: {len(required_ids)}")
    
except Exception as e:
    print(f"Ошибка чтения датасета: {e}")
    # Если нет датасета, предполагаем ID от 0 до 1004
    required_ids = list(range(0, 1005))
    print("Используем стандартный диапазон ID: 0-1004")

print(f"Требуется ID: от {min(required_ids)} до {max(required_ids)} (всего {len(required_ids)})")

# Читаем submission файл
try:
    df_submission = pd.read_csv('submission.csv')
    print(f"\n=== АНАЛИЗ SUBMISSION ФАЙЛА ===")
    print(f"Submission содержит {len(df_submission)} строк")
    print(f"ID в submission: от {df_submission['id'].min()} до {df_submission['id'].max()}")
    
    # Показываем примеры текущего формата
    print("\nПримеры текущего формата predicted_positions:")
    for i in range(min(5, len(df_submission))):
        print(f"  ID {df_submission.iloc[i]['id']}: {repr(df_submission.iloc[i]['predicted_positions'])}")
        
except Exception as e:
    print(f"Ошибка чтения submission файла: {e}")
    print("Создаем файл с пустыми массивами для всех ID")
    df_submission = pd.DataFrame({
        'id': required_ids,
        'predicted_positions': ['[]'] * len(required_ids)
    })

# Создаем исправленный файл
print(f"\n=== ИСПРАВЛЕНИЕ ===")

# Создаем DataFrame с правильными ID
df_fixed = pd.DataFrame({'id': required_ids})

if len(df_submission) > 0:
    # Подготавливаем submission данные
    df_sub_clean = df_submission.copy()
    df_sub_clean['id'] = df_sub_clean['id'].astype(int)
    
    # Исправляем формат predicted_positions
    df_sub_clean['predicted_positions'] = df_sub_clean['predicted_positions'].astype(str)
    # Убираем кавычки, но оставляем строковый тип
    df_sub_clean['predicted_positions'] = df_sub_clean['predicted_positions'].str.replace('"', '')
    
    # Объединяем с полным списком ID
    df_fixed = df_fixed.merge(df_sub_clean[['id', 'predicted_positions']], 
                             on='id', how='left')
    
    print("Объединение выполнено")
else:
    df_fixed['predicted_positions'] = None

# Заполняем пропущенные значения пустыми массивами
df_fixed['predicted_positions'] = df_fixed['predicted_positions'].fillna('[]')

print(f"Итоговое количество строк: {len(df_fixed)}")

# Статистика
missing_count = (df_fixed['predicted_positions'] == '[]').sum()
print(f"Строк с пустыми массивами: {missing_count}")

# Показываем результат
print(f"\n=== РЕЗУЛЬТАТ ===")
print("Первые 10 строк:")
print(df_fixed.head(10))

print("\nПоследние 5 строк:")
print(df_fixed.tail(5))

# Валидация формата
print(f"\n=== ВАЛИДАЦИЯ ===")
print(f"✓ Правильное количество строк: {len(df_fixed) == len(required_ids)}")
print(f"✓ Все ID присутствуют: {set(df_fixed['id']) == set(required_ids)}")
print(f"✓ Нет дубликатов: {len(df_fixed['id'].unique()) == len(df_fixed)}")

# Проверяем что можно парсить как массивы
validation_errors = 0
for i in range(min(10, len(df_fixed))):
    try:
        pos_str = df_fixed.iloc[i]['predicted_positions']
        parsed = ast.literal_eval(pos_str)
        if not isinstance(parsed, list):
            validation_errors += 1
            print(f"  Ошибка: строка {i} не является списком: {pos_str}")
    except Exception as e:
        validation_errors += 1
        print(f"  Ошибка парсинга строки {i}: {pos_str} - {e}")

if validation_errors == 0:
    print("✓ Все проверенные значения корректны")
else:
    print(f"❌ Найдено ошибок: {validation_errors}")

# Сохраняем
output_filename = 'submission_fixed_final.csv'
df_fixed.to_csv(output_filename, index=False)
print(f"\n✅ Исправленный файл сохранен как '{output_filename}'")

# Финальная проверка сохраненного файла
df_check = pd.read_csv(output_filename)
print(f"\n=== ПРОВЕРКА СОХРАНЕННОГО ФАЙЛА ===")
print(f"Загружено строк: {len(df_check)}")
print("Типы данных:")
print(df_check.dtypes)
print("\nПример содержимого:")
print(df_check.head(3))

# Создаем версии с разными форматами кавычек
print(f"\n=== СОЗДАНИЕ АЛЬТЕРНАТИВНЫХ ВЕРСИЙ ===")

# Версия 1: БЕЗ кавычек (как основной файл)
with open('submission_fixed_no_quotes.csv', 'w', encoding='utf-8') as f:
    f.write('id,predicted_positions\n')
    for _, row in df_fixed.iterrows():
        positions = str(row['predicted_positions']).replace('"', '')
        f.write(f"{row['id']},{positions}\n")

# Версия 2: ВСЕ значения predicted_positions В КАВЫЧКАХ
with open('submission_fixed_with_quotes.csv', 'w', encoding='utf-8') as f:
    f.write('id,predicted_positions\n')
    for _, row in df_fixed.iterrows():
        positions = str(row['predicted_positions']).replace('"', '')  # Убираем существующие кавычки
        f.write(f'{row["id"]},"{positions}"\n')  # Добавляем кавычки вокруг значения

print("✅ Создан файл БЕЗ кавычек: 'submission_fixed_no_quotes.csv'")
print("✅ Создан файл С кавычками: 'submission_fixed_with_quotes.csv'")

# Показываем примеры разных форматов
print("\n=== СРАВНЕНИЕ ФОРМАТОВ ===")
print("Формат БЕЗ кавычек:")
with open('submission_fixed_no_quotes.csv', 'r', encoding='utf-8') as f:
    lines = f.readlines()
    for i in range(min(5, len(lines))):
        print(f"  {lines[i].strip()}")

print("\nФормат С кавычками:")
with open('submission_fixed_with_quotes.csv', 'r', encoding='utf-8') as f:
    lines = f.readlines()
    for i in range(min(5, len(lines))):
        print(f"  {lines[i].strip()}")

print(f"\n🎯 ИТОГ:")
print(f"1. Основной файл: {output_filename}")
print(f"2. БЕЗ кавычек: submission_fixed_no_quotes.csv")
print(f"3. С КАВЫЧКАМИ: submission_fixed_with_quotes.csv")
print(f"4. Количество строк: {len(df_fixed)}")
print(f"5. Диапазон ID: {min(required_ids)} - {max(required_ids)}")
print("\nФорматы:")
print('  - Без кавычек: 0,[1,5,10]')
print('  - С кавычками: 0,"[1,5,10]"')
print("\n📋 РЕКОМЕНДАЦИИ:")
print("1. Попробуйте сначала файл С КАВЫЧКАМИ: submission_fixed_with_quotes.csv")
print("2. Если не примет - попробуйте без кавычек: submission_fixed_no_quotes.csv")
print("3. Оба файла содержат правильные ID от 0 до 1004 (1005 строк)")
print("4. Пропущенные ID заполнены пустыми массивами []")