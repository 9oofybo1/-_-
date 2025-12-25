import sys
import os
from PyQt5.QtWidgets import QApplication, QMessageBox
from main_window import MainWindow
from database import init_db
from recognition import face_recognizer, check_model_status


def initialize_application():
    """Инициализирует все компоненты приложения"""
    print("=" * 50)
    print("ЗАПУСК СИСТЕМЫ РАСПОЗНАВАНИЯ ЛИЦ")
    print("=" * 50)

    # 1. Инициализация базы данных
    print("1. Инициализация базы данных...")
    init_db()
    print("   ✓ База данных готова")

    # 2. Проверка и загрузка модели
    print("2. Проверка модели распознавания...")

    # Сначала пробуем загрузить сохраненную модель
    model_loaded = False
    if os.path.exists("face_recognition_model.yml"):
        print("   Найдена сохраненная модель, загружаю...")
        model_loaded = face_recognizer.load_model()

    # Если модель не загружена или не обучена, показываем предупреждение
    if not model_loaded or not check_model_status():
        print("   ⚠️ Модель не обучена или повреждена")
        print("   Запустите обучение через меню или train_model.py")

        # Создаем QApplication для показа диалога
        app_temp = QApplication(sys.argv)

        reply = QMessageBox.question(
            None, 'Модель не обучена',
            'Модель распознавания лиц не обучена.\n\n'
            '1. Обучить сейчас (займет 10-30 секунд)\n'
            '2. Продолжить без распознавания\n\n'
            'Обучить модель?',
            QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel,
            QMessageBox.Yes
        )

        if reply == QMessageBox.Yes:
            # Обучаем модель
            from database import get_all_photos, get_all_persons
            all_photos = get_all_photos()
            all_persons = get_all_persons()

            if all_photos:
                print("   Обучение модели...")
                success = face_recognizer.train(all_photos, all_persons)
                if success:
                    print("   ✓ Модель успешно обучена")
                    face_recognizer.save_model()
                else:
                    print("   ✗ Ошибка обучения модели")
            else:
                QMessageBox.warning(
                    None, 'Нет данных',
                    'В базе нет фотографий для обучения!\n\n'
                    'Добавьте людей через раздел "Добавить человека"'
                )
        elif reply == QMessageBox.Cancel:
            print("   Запуск отменен")
            sys.exit(0)
        else:
            print("   Продолжаю без обученной модели")

    else:
        print(f"   ✓ Модель загружена ({len(face_recognizer.labels)} лиц)")

    print("3. Запуск графического интерфейса...")
    return True


def main():
    # Инициализируем приложение
    if not initialize_application():
        print("Не удалось инициализировать приложение")
        return

    # Создаем основное приложение
    app = QApplication(sys.argv)

    # Создание главного окна
    window = MainWindow()
    window.show()

    # Запускаем статус модели в заголовок окна
    if check_model_status():
        window.setWindowTitle(f"Facer - Распознавание лиц ({len(face_recognizer.labels)} лиц в базе)")
    else:
        window.setWindowTitle("Facer - Модель не обучена")

    print("\n✅ ПРИЛОЖЕНИЕ ЗАПУЩЕНО")
    print("=" * 50)

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()