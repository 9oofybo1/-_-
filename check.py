# save_as: check_state.py
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

print("=== ДИАГНОСТИКА СИСТЕМЫ ===")

# 1. Проверка файлов
print("\n1. Файлы проекта:")
files = [
    "main.py", "main_window.py", "recognition.py",
    "recognition_service.py", "database.py", "image_utils.py",
    "face_recognition_model.yml"
]

for f in files:
    exists = os.path.exists(f)
    size = os.path.getsize(f) if exists else 0
    print(f"   {f}: {'✓' if exists else '✗'} ({size/1024:.1f} KB)")

# 2. Попробуем импортировать
print("\n2. Проверка импортов:")
try:
    from database import init_db, get_all_photos
    print("   database: ✓")
except Exception as e:
    print(f"   database: ✗ ({e})")

try:
    from recognition import face_recognizer
    print("   recognition: ✓")
except Exception as e:
    print(f"   recognition: ✗ ({e})")

# 3. Проверка базы
print("\n3. Проверка базы данных:")
try:
    init_db()
    photos = get_all_photos()
    print(f"   Фото в базе: {len(photos)}")
except Exception as e:
    print(f"   Ошибка базы: {e}")

# 4. Проверка модели
print("\n4. Проверка модели:")
if os.path.exists("face_recognition_model.yml"):
    try:
        if face_recognizer.load_model():
            print(f"   Модель загружена")
            print(f"   is_trained: {face_recognizer.is_trained}")
            print(f"   Лиц в модели: {len(face_recognizer.labels)}")
        else:
            print("   Не удалось загрузить модель")
    except Exception as e:
        print(f"   Ошибка загрузки: {e}")
else:
    print("   Файл модели не найден")

print("\n=== РЕКОМЕНДАЦИИ ===")
if not os.path.exists("face_recognition_model.yml"):
    print("1. Запустите: python train_model.py")
elif face_recognizer.is_trained and len(face_recognizer.labels) > 0:
    print("1. Модель готова, запускайте: python main.py")
else:
    print("1. Нужно обучить модель")

input("\nНажмите Enter для выхода...")