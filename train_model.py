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

    # Обучаем модель
    print("\nНачинаю обучение модели LBPH...")
    success = face_recognizer.train(all_photos, all_persons)

    if success:
        print(f"\n✅ МОДЕЛЬ УСПЕШНО ОБУЧЕНА!")
        print(f"   Обучено на {len(face_recognizer.labels)} уникальных лицах")

        # Сохраняем модель
        face_recognizer.save_model()

        # Тестируем модель
        print("\nТестирование модели:")
        print(f"Статус модели: {'Обучена' if face_recognizer.is_trained else 'Не обучена'}")
        print(f"Количество лиц: {len(face_recognizer.labels)}")

        # Тестируем сравнение лиц
        if all_photos:
            try:
                import cv2
                import numpy as np

                # Берем первое фото для теста
                person_id, blob = all_photos[0]
                test_face = cv2.imdecode(np.frombuffer(blob, np.uint8), cv2.IMREAD_GRAYSCALE)

                # Тестируем предсказание
                label, confidence, name = face_recognizer.predict(test_face)
                similarity = max(0, 100 - confidence)

                print(f"\nТестовое предсказание:")
                print(f"  ID: {label}, Сходство: {similarity:.1f}%, Имя: {name}")
            except Exception as e:
                print(f"Ошибка тестирования: {e}")

        return True
    else:
        print("\n❌ ОШИБКА ОБУЧЕНИЯ МОДЕЛИ!")
        return False


if __name__ == "__main__":
    success = train_model()

    if success:
        print("\n" + "=" * 50)
        print("Модель готова к использованию!")
        print("Запустите приложение: python main.py")
    else:
        print("\n" + "=" * 50)
        print("Не удалось обучить модель!")

    input("\nНажмите Enter для выхода...")