import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from database import init_db, get_all_photos, get_all_persons
from recognition import face_recognizer, check_model_status

# Инициализируем базу
init_db()

print("=== ДИАГНОСТИКА МОДЕЛИ РАСПОЗНАВАНИЯ ===")
print()

# 1. Проверяем данные в базе
all_photos = get_all_photos()
all_persons = get_all_persons()

print(f"1. Данные в базе:")
print(f"   - Людей: {len(all_persons)}")
print(f"   - Фотографий: {len(all_photos)}")
print()

# 2. Проверяем состояние модели
print(f"2. Состояние модели:")
print(f"   - is_trained: {face_recognizer.is_trained}")
print(f"   - Количество лиц: {len(face_recognizer.labels)}")
print(f"   - check_model_status(): {check_model_status()}")
print()

# 3. Пытаемся загрузить модель
print(f"3. Загрузка модели:")
model_loaded = face_recognizer.load_model()
print(f"   - load_model() результат: {model_loaded}")
print(f"   - После загрузки is_trained: {face_recognizer.is_trained}")
print(f"   - После загрузки количество лиц: {len(face_recognizer.labels)}")
print()

# 4. Если модель не загружена, пробуем обучить
if not face_recognizer.is_trained and all_photos:
    print(f"4. Пробуем обучить модель...")
    success = face_recognizer.train(all_photos, all_persons)
    print(f"   - train() результат: {success}")
    print(f"   - После обучения is_trained: {face_recognizer.is_trained}")
    print(f"   - После обучения количество лиц: {len(face_recognizer.labels)}")

    if success:
        face_recognizer.save_model()
        print(f"   - Модель сохранена")
print()

# 5. Тестируем функцию compare_faces
print(f"5. Тестируем compare_faces:")
if face_recognizer.is_trained and all_photos:
    try:
        # Берем первое фото из базы
        person_id, first_blob = all_photos[0]

        # Декодируем фото
        import cv2
        import numpy as np

        test_face = cv2.imdecode(np.frombuffer(first_blob, np.uint8), cv2.IMREAD_GRAYSCALE)

        # Пробуем сравнить с самим собой
        score = face_recognizer.compare_faces(test_face, test_face)
        print(f"   - compare_faces(то_же_лицо, то_же_лицо): {score:.1f}%")

        if len(all_photos) > 1:
            # Берем фото другого человека если есть
            person_id2, second_blob = all_photos[1]
            if person_id != person_id2:
                test_face2 = cv2.imdecode(np.frombuffer(second_blob, np.uint8), cv2.IMREAD_GRAYSCALE)
                score2 = face_recognizer.compare_faces(test_face, test_face2)
                print(f"   - compare_faces(лицо1, лицо2): {score2:.1f}%")
    except Exception as e:
        print(f"   - Ошибка тестирования: {e}")
else:
    print(f"   - Модель не обучена, тест пропущен")

print()
print("=== КОНЕЦ ДИАГНОСТИКИ ===")
input("Нажмите Enter для выхода...")