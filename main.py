import sys
from PyQt5.QtWidgets import QApplication
from main_window import MainWindow
from database import init_db
from recognition import face_recognizer
from database import get_all_photos, get_all_persons


def initialize_face_recognition():
    """Инициализирует систему распознавания лиц"""
    print("Инициализация системы распознавания...")

    # Инициализация базы данных
    init_db()

    # Пробуем загрузить модель
    model_loaded = face_recognizer.load_model()

    if not model_loaded:
        print("Модель не найдена, обучаю новую...")
        # Получаем данные для обучения
        all_photos = get_all_photos()
        all_persons = get_all_persons()

        if not all_photos:
            print("В базе нет фотографий для обучения модели")
            return False

        # Обучаем модель
        success = face_recognizer.train(all_photos, all_persons)

        if success:
            print(f"Модель обучена успешно! Лиц в базе: {len(face_recognizer.labels)}")
            face_recognizer.save_model()
            return True
        else:
            print("Ошибка обучения модели")
            return False

    print(f"Модель загружена. Лиц в базе: {len(face_recognizer.labels)}")
    return True


def main():
    # Инициализация системы распознавания
    if not initialize_face_recognition():
        print("Предупреждение: система распознавания не инициализирована")

    # Создание приложения
    app = QApplication(sys.argv)

    # Создание главного окна
    window = MainWindow()
    window.show()

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()