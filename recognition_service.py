"""
Сервис для распознавания лиц с использованием обученной модели
"""
import cv2
import numpy as np
from image_utils import extract_face
from recognition import face_recognizer
from database import get_all_photos, get_person_by_id, add_recognition_log, get_all_persons

THRESHOLD = 70.0  # Повышенный порог уверенности (в процентах)


def initialize_recognition():
    """
    Инициализирует систему распознавания
    Загружает модель или обучает новую если нет сохраненной
    """
    print("Инициализация системы распознавания...")

    # Пытаемся загрузить сохраненную модель
    if face_recognizer.load_model():
        print("✓ Модель распознавания загружена")
        return True

    # Если нет сохраненной модели, обучаем новую
    print("Сохраненной модели нет. Обучаю новую...")
    all_photos = get_all_photos()
    all_persons = get_all_persons()

    if not all_photos:
        print("✗ В базе нет фотографий для обучения")
        return False

    success = face_recognizer.train(all_photos, all_persons)

    if success:
        print("✓ Модель успешно обучена")
        return True
    else:
        print("✗ Не удалось обучить модель")
        return False


def retrain_model():
    """
    Переобучает модель на текущих данных из базы
    Используется при добавлении/удалении людей
    """
    print("Переобучение модели...")
    all_photos = get_all_photos()
    all_persons = get_all_persons()

    if not all_photos:
        print("Нет данных для обучения")
        return False

    return face_recognizer.train(all_photos, all_persons)


def recognize_face(face_image):
    """
    Распознает лицо на изображении

    Args:
        face_image: изображение лица в оттенках серого

    Returns:
        dict: {
            'person_id': ID человека или None,
            'similarity': процент сходства (0-100),
            'person_name': имя человека или строка ошибки,
            'recognized': True если распознан успешно
        }
    """
    if not face_recognizer.is_trained:
        return {
            'person_id': None,
            'similarity': 0,
            'person_name': 'Модель не обучена',
            'recognized': False
        }

    # Распознаем лицо
    person_id, similarity, person_name = face_recognizer.predict(face_image)

    # Проверяем порог
    recognized = similarity >= THRESHOLD and person_id is not None

    # Логируем результат
    try:
        result = "SUCCESS" if recognized else "FAILED"
        add_recognition_log(person_id if recognized else None, similarity, result)
    except:
        pass  # Не прерываем работу если логирование не удалось

    return {
        'person_id': person_id,
        'similarity': similarity,
        'person_name': person_name,
        'recognized': recognized
    }


def recognize_from_camera():
    """
    Функция для распознавания с камеры (консольный режим)
    Сохраняется для обратной совместимости
    """
    # Инициализируем распознавание
    if not initialize_recognition():
        print("Не удалось инициализировать распознавание")
        return

    cap = cv2.VideoCapture(0)
    print("ESC — выход")

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        face = extract_face(frame)
        result_text = "ЛИЦО НЕ ОБНАРУЖЕНО"

        if face is not None:
            # Распознаем лицо
            result = recognize_face(face)

            if result['recognized']:
                person = get_person_by_id(result['person_id'])
                if person:
                    result_text = f"{person[1]} {person[2]} ({result['similarity']:.1f}%)"
                else:
                    result_text = f"{result['person_name']} ({result['similarity']:.1f}%)"
            else:
                result_text = f"НЕОПОЗНАННОЕ ЛИЦО ({result['similarity']:.1f}%)"

        # Отображаем результат на кадре
        cv2.putText(
            frame, result_text, (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1, (0, 255, 0) if "НЕОПОЗНАННО" not in result_text else (0, 0, 255), 2
        )

        cv2.imshow("Face Recognition", frame)

        if cv2.waitKey(1) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

# Автоматически инициализируем при импорте (можно отключить)
# initialize_recognition()