"""
Модуль для сравнения лиц с использованием LBPH-распознавателя.
LBPH (Local Binary Patterns Histograms) устойчив к изменениям освещения.
"""
import cv2
import numpy as np
import pickle
import os

MODEL_FILE = "face_recognition_model.yml"


class FaceRecognizer:
    """Класс для распознавания лиц с использованием LBPH"""

    def __init__(self):
        # ПРОСТЫЕ параметры LBPH (минимум вычислений)
        self.recognizer = cv2.face.LBPHFaceRecognizer_create(
            radius=1,  # Меньше вычислений
            neighbors=8,  # Минимум соседей
            grid_x=7,  # Упрощенная сетка
            grid_y=7,
            threshold=100  # Стандартный порог
        )
        self.labels = []
        self.label_names = {}
        self.is_trained = False

    def prepare_training_data(self, all_photos, all_persons):
        """
        Подготавливает данные для обучения с контролем памяти
        """
        faces = []
        labels = []

        self.label_names = {p[0]: f"{p[1]} {p[2]}" for p in all_persons}

        total = len(all_photos)
        print(f"Подготовка {total} фото для обучения...")

        for idx, (person_id, blob) in enumerate(all_photos):
            try:
                # Показываем прогресс каждые 50 фото
                if idx % 50 == 0:
                    print(f"  Обработано {idx}/{total} фото")

                # Декодируем изображение
                img = cv2.imdecode(
                    np.frombuffer(blob, np.uint8),
                    cv2.IMREAD_GRAYSCALE
                )

                if img is not None:
                    # МИНИМАЛЬНАЯ обработка
                    img = cv2.resize(img, (200, 200))
                    faces.append(img)
                    labels.append(person_id)

                # Очищаем память каждые 100 фото
                if idx % 100 == 0:
                    import gc
                    gc.collect()

            except Exception as e:
                if idx % 20 == 0:  # Не спамить ошибками
                    print(f"Ошибка фото {idx}: {e}")
                continue

        print(f"Подготовлено {len(faces)} лиц из {total} фото")
        return faces, labels

    def train(self, all_photos, all_persons, progress_callback=None):
        """
        Обучает модель на данных из базы с возможностью отслеживания прогресса

        Args:
            all_photos: список фотографий
            all_persons: список людей
            progress_callback: функция для отслеживания прогресса (опционально)
        """
        if not all_photos:
            print("Нет данных для обучения")
            return False

        print(f"Начинаю обучение модели на {len(all_photos)} фото...")

        try:
            # Подготавливаем данные с прогрессом
            total = len(all_photos)
            faces = []
            labels = []

            # Создаем mapping ID -> имя для удобства
            self.label_names = {p[0]: f"{p[1]} {p[2]}" for p in all_persons}

            for i, (person_id, blob) in enumerate(all_photos):
                try:
                    # Обновляем прогресс каждые 10 фото
                    if progress_callback and i % 10 == 0:
                        progress = int((i / total) * 50)  # 50% на подготовку
                        progress_callback(progress, f"Обработка фото {i + 1}/{total}")

                    # Декодируем изображение из blob
                    img = cv2.imdecode(
                        np.frombuffer(blob, np.uint8),
                        cv2.IMREAD_GRAYSCALE
                    )

                    if img is not None:
                        # Препроцессинг
                        img = self.preprocess_face(img)
                        img = cv2.resize(img, (200, 200))
                        faces.append(img)
                        labels.append(person_id)

                except Exception as e:
                    print(f"Ошибка обработки фото {i + 1}: {e}")
                    continue

            print(f"Подготовлено {len(faces)} лиц для обучения")

            if not faces:
                print("Не удалось подготовить лица для обучения")
                return False

            # Обучаем модель
            if progress_callback:
                progress_callback(75, "Обучение модели LBPH...")

            self.recognizer.train(faces, np.array(labels))
            self.labels = list(set(labels))
            self.is_trained = True

            if progress_callback:
                progress_callback(100, "Обучение завершено")

            print(f"Модель обучена успешно! Уникальных лиц: {len(self.labels)}")

            # Сохраняем модель
            self.save_model()

            return True

        except Exception as e:
            print(f"Ошибка обучения модели: {e}")
            import traceback
            traceback.print_exc()
            return False

    def predict(self, face_image):
        """
        Распознает лицо на изображении
        """
        if not self.is_trained:
            return None, 1000, "Модель не обучена"

        try:
            # ПРЕПРОЦЕССИНГ перед распознаванием
            face_image = self.preprocess_face(face_image)

            # Нормализуем размер
            face_resized = cv2.resize(face_image, (200, 200))

            # Предсказываем
            label, confidence = self.recognizer.predict(face_resized)

            # В LBPH confidence - это расстояние (чем меньше, тем лучше)
            # Преобразуем в проценты (0-100, где 100 - лучше)
            similarity_score = max(0, 100 - confidence)

            # Получаем имя человека
            person_name = self.label_names.get(label, f"ID {label}")

            return label, similarity_score, person_name

        except Exception as e:
            print(f"Ошибка предсказания: {e}")
            return None, 0, "Ошибка распознавания"

    def save_model(self, filename=MODEL_FILE):
        """Сохраняет модель в файл"""
        try:
            self.recognizer.write(filename)

            # Сохраняем дополнительные данные
            metadata = {
                'labels': self.labels,
                'label_names': self.label_names,
                'is_trained': self.is_trained
            }

            with open(filename + '.meta', 'wb') as f:
                pickle.dump(metadata, f)

            print(f"Модель сохранена в {filename}")
            return True

        except Exception as e:
            print(f"Ошибка сохранения модели: {e}")
            return False

    def load_model(self, filename=MODEL_FILE):
        """Загружает модель из файла"""
        try:
            if not os.path.exists(filename):
                print(f"Файл модели {filename} не найден")
                return False

            # Загружаем модель
            self.recognizer.read(filename)

            # Загружаем метаданные
            meta_file = filename + '.meta'
            if os.path.exists(meta_file):
                with open(meta_file, 'rb') as f:
                    metadata = pickle.load(f)
                    self.labels = metadata.get('labels', [])
                    self.label_names = metadata.get('label_names', {})
                    self.is_trained = metadata.get('is_trained', False)

            print(f"Модель загружена из {filename}. Лиц в базе: {len(self.labels)}")
            return True

        except Exception as e:
            print(f"Ошибка загрузки модели: {e}")
            return False

    def compare_faces(self, face1, face2):
        """
        Сравнивает два лица напрямую (для обратной совместимости)

        Args:
            face1, face2: изображения лиц

        Returns:
            float: процент сходства (0-100)
        """
        if not self.is_trained:
            return 0

        try:
            # Создаем временную модель для сравнения двух лиц
            temp_recognizer = cv2.face.LBPHFaceRecognizer_create()

            # Подготавливаем данные
            faces = [
                cv2.resize(face1, (200, 200)),
                cv2.resize(face2, (200, 200))
            ]
            labels = [1, 2]  # Разные метки

            # Обучаем на двух лицах
            temp_recognizer.train(faces, np.array(labels))

            # Пытаемся распознать первое лицо как второе
            test_face = cv2.resize(face1, (200, 200))
            label, confidence = temp_recognizer.predict(test_face)

            # Преобразуем confidence в проценты
            similarity = max(0, 100 - confidence)

            return similarity

        except Exception as e:
            print(f"Ошибка сравнения лиц: {e}")
            return 0

    def preprocess_face(self, face_image):
        """
        Подготавливает лицо для лучшего распознавания
        """
        if face_image is None or face_image.size == 0:
            return face_image

        try:
            # 1. Нормализация гистограммы (выравнивание освещения)
            face_image = cv2.equalizeHist(face_image)

            # 2. Гауссово размытие для уменьшения шума
            face_image = cv2.GaussianBlur(face_image, (3, 3), 0)

            # 3. Увеличение контраста
            face_image = cv2.convertScaleAbs(face_image, alpha=1.2, beta=10)

            return face_image
        except Exception as e:
            print(f"Ошибка препроцессинга: {e}")
            return face_image

# Создаем глобальный экземпляр для использования в других модулях
face_recognizer = FaceRecognizer()

def compare_faces(face1, face2):
    """
    Функция для обратной совместимости со старым кодом
    Использует новую модель LBPH для сравнения
    """
    return face_recognizer.compare_faces(face1, face2)

def check_model_status():
    """
    Проверяет состояние модели распознавания
    Возвращает True если модель обучена и готова к работе
    """
    return face_recognizer.is_trained and len(face_recognizer.labels) > 0

def get_model_info():
    """
    Возвращает информацию о состоянии модели
    """
    return {
        'is_trained': face_recognizer.is_trained,
        'num_faces': len(face_recognizer.labels),
        'label_names': face_recognizer.label_names
    }

# Функция для обратной совместимости
def compare_faces(face1, face2):
    """
    Функция для обратной совместимости со старым кодом
    """
    return face_recognizer.compare_faces(face1, face2)

# Функция для проверки состояния модели
def check_model_status():
    """Проверяет состояние модели распознавания"""
    return face_recognizer.is_trained and len(face_recognizer.labels) > 0

def get_model_info():
    """Возвращает информацию о состоянии модели"""
    return {
        'is_trained': face_recognizer.is_trained,
        'num_faces': len(face_recognizer.labels),
        'label_names': face_recognizer.label_names
    }