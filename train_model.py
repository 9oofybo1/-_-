#!/usr/bin/env python3
"""
Скрипт для обучения модели распознавания лиц
"""
import sys
import os

# Добавляем текущую директорию в путь для импортов
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from database import init_db, get_all_photos, get_all_persons
from recognition import face_recognizer


def train_model():
    """Обучает модель на данных из базы"""
    print("=" * 50)
    print("ОБУЧЕНИЕ МОДЕЛИ РАСПОЗНАВАНИЯ ЛИЦ")
    print("=" * 50)

    # Инициализируем базу данных
    init_db()

    # Получаем данные из базы
    all_photos = get_all_photos()
    all_persons = get_all_persons()

    print(f"Найдено в базе:")
    print(f"  - Людей: {len(all_persons)}")
    print(f"  - Фотографий: {len(all_photos)}")

    if not all_photos:
        print("ОШИБКА: В базе нет фотографий для обучения!")
        print("Добавьте людей через интерфейс сначала.")
        return False

    # Выводим информацию о людях
    print("\nЛюди в базе:")
    for person in all_persons:
        person_id, first_name, last_name, group = person
        # Считаем фото для этого человека
        photo_count = sum(1 for pid, _ in all_photos if pid == person_id)
        print(f"  ID {person_id}: {first_name} {last_name} ({group}) - {photo_count} фото")

    # Обучаем модель
    print("\nНачинаю обучение модели LBPH...")
    success = face_recognizer.train(all_photos, all_persons)

    if success:
        print("\n✅ МОДЕЛЬ УСПЕШНО ОБУЧЕНА!")
        print(f"   Обучено на {len(face_recognizer.labels)} уникальных лицах")

        # Сохраняем модель
        face_recognizer.save_model()

        # Тестируем модель
        print("\nТестирование модели:")
        print("Статус модели:", "Обучена" if face_recognizer.is_trained else "Не обучена")
        print("Количество лиц:", len(face_recognizer.labels))

        return True
    else:
        print("\n❌ ОШИБКА ОБУЧЕНИЯ МОДЕЛИ!")
        return False


if __name__ == "__main__":
    train_model()

    # Ждем нажатия Enter перед закрытием
    input("\nНажмите Enter для выхода...")