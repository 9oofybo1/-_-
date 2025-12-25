import sys
from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFrame, QStackedWidget, QTextEdit,
    QTableView, QMessageBox, QSizePolicy, QGridLayout, QSpacerItem, QProgressBar
)
from PyQt5.QtCore import Qt, pyqtSignal, QTimer, QSize
from PyQt5.QtGui import QFont, QIcon, QImage, QPixmap, QStandardItemModel, QStandardItem
import cv2
import numpy as np

from styles import STYLE
from image_utils import extract_face
from recognition import compare_faces
from database import get_all_photos, get_person_by_id, get_all_persons, delete_person, get_photos_by_person


class CameraManager:
    """Менеджер для управления камерой"""

    _instance = None
    _camera = None
    _timer = None
    _is_running = False
    _current_user = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(CameraManager, cls).__new__(cls)
        return cls._instance

    def start_camera(self, user_id):
        """Запускает камеру для конкретного пользователя"""
        # Если камера уже запущена для другого пользователя, останавливаем её
        if self._current_user is not None and self._current_user != user_id:
            self.stop_camera()

        # Если камера не запущена, запускаем
        if not self._is_running:
            self._camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)
            if not self._camera.isOpened():
                print("Ошибка: не удалось открыть камеру")
                return False
            self._is_running = True

        self._current_user = user_id
        return True

    def stop_camera(self, user_id=None):
        """Останавливает камеру"""
        # Если указан user_id, останавливаем только если это текущий пользователь
        if user_id is not None and self._current_user != user_id:
            return

        if self._is_running and self._camera is not None:
            if self._camera.isOpened():
                self._camera.release()
            self._camera = None
            self._is_running = False
            self._current_user = None

    def is_camera_available(self, user_id):
        """Проверяет, доступна ли камера для пользователя"""
        return self._is_running and self._current_user == user_id

    def get_frame(self):
        """Получает текущий кадр с камеры"""
        if not self._is_running or self._camera is None:
            return None

        ret, frame = self._camera.read()
        if not ret:
            return None
        return frame

    def is_opened(self):
        """Проверяет, открыта ли камера"""
        return self._is_running and self._camera is not None and self._camera.isOpened()


class NavigationButton(QPushButton):
    """Кнопка для навигации"""

    def __init__(self, text, icon_text="", parent=None):
        super().__init__(text, parent)
        self.setObjectName("navButton")
        self.setCheckable(True)
        self.setFixedHeight(50)
        self.setCursor(Qt.PointingHandCursor)
        if icon_text:
            self.setText(f"  {icon_text}  {text}")


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Система распознавания лиц")
        self.setMinimumSize(1500, 1000)
        self.setStyleSheet(STYLE)

        # Создаем менеджер камеры
        self.camera_manager = CameraManager()

        # Центральный виджет
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Основной layout
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # Левая панель навигации
        self.create_navigation_panel(main_layout)

        # Область контента
        self.create_content_area(main_layout)

        # Инициализация страниц
        self.init_pages()

    def create_navigation_panel(self, parent_layout):
        """Создает левую панель навигации"""
        nav_frame = QFrame()
        nav_frame.setObjectName("navFrame")
        nav_frame.setFixedWidth(250)

        nav_layout = QVBoxLayout(nav_frame)
        nav_layout.setContentsMargins(10, 30, 10, 30)
        nav_layout.setSpacing(10)

        nav_layout.addSpacing(30)

        # Кнопки навигации
        self.recognition_btn = NavigationButton("Распознавание", "🎥")
        self.database_btn = NavigationButton("База данных", "👥")
        self.add_person_btn = NavigationButton("Добавить человека", "➕")

        # Подключаем кнопки
        self.recognition_btn.clicked.connect(lambda: self.switch_page(0))
        self.database_btn.clicked.connect(lambda: self.switch_page(1))
        self.add_person_btn.clicked.connect(lambda: self.switch_page(2))

        nav_layout.addWidget(self.recognition_btn)
        nav_layout.addWidget(self.database_btn)
        nav_layout.addWidget(self.add_person_btn)

        nav_layout.addStretch()

        # Кнопка выхода
        exit_btn = QPushButton("Выход", self)
        exit_btn.clicked.connect(self.close)
        exit_btn.setObjectName("navButton")
        nav_layout.addWidget(exit_btn)

        parent_layout.addWidget(nav_frame)

    def create_content_area(self, parent_layout):
        """Создает область контента со страницами"""
        content_frame = QFrame()
        content_layout = QVBoxLayout(content_frame)
        content_layout.setContentsMargins(30, 30, 30, 30)
        content_layout.setSpacing(20)

        # Создаем StackedWidget для переключения страниц
        self.stacked_widget = QStackedWidget()

        # Создаем контейнеры для страниц
        self.recognition_page = QWidget()
        self.recognition_page.setObjectName("recognitionPage")

        self.database_page = QWidget()
        self.database_page.setObjectName("databasePage")

        self.add_person_page = QWidget()
        self.add_person_page.setObjectName("addPersonPage")

        # Добавляем страницы
        self.stacked_widget.addWidget(self.recognition_page)
        self.stacked_widget.addWidget(self.database_page)
        self.stacked_widget.addWidget(self.add_person_page)

        content_layout.addWidget(self.stacked_widget)
        parent_layout.addWidget(content_frame, stretch=1)

    def init_pages(self):
        """Инициализирует содержимое страниц из старых окон"""
        # Страница распознавания
        self.init_recognition_page()

        # Страница базы данных
        self.init_database_page()

        # Страница добавления человека
        self.init_add_person_page()

        # Активируем первую страницу
        self.recognition_btn.setChecked(True)

    def init_recognition_page(self):
        """Инициализация страницы распознавания из старого RecognitionWindow"""
        old_window = RecognitionWindow(self.camera_manager, "recognition")

        # Удаляем настройки окна
        old_window.setWindowFlags(Qt.Widget)
        old_window.setParent(self.recognition_page)

        # Создаем layout для страницы
        layout = QVBoxLayout(self.recognition_page)
        layout.setContentsMargins(0, 0, 0, 0)

        # Добавляем виджет распознавания
        layout.addWidget(old_window)

        # Сохраняем ссылку для доступа
        self.recognition_widget = old_window

    def init_database_page(self):
        """Инициализация страницы базы данных из старого DatabaseWindow"""
        old_window = DatabaseWindow()

        # Удаляем настройки окна
        old_window.setWindowFlags(Qt.Widget)
        old_window.setParent(self.database_page)

        # Создаем layout для страницы
        layout = QVBoxLayout(self.database_page)
        layout.setContentsMargins(0, 0, 0, 0)

        # Добавляем виджет базы данных
        layout.addWidget(old_window)

        # Сохраняем ссылку для доступа
        self.database_widget = old_window

    def init_add_person_page(self):
        """Инициализация страницы добавления человека"""
        add_person_widget = AddPersonWindow(self.camera_manager, "add_person")

        # Создаем layout для страницы
        layout = QVBoxLayout(self.add_person_page)
        layout.setContentsMargins(0, 0, 0, 0)

        # Добавляем виджет добавления
        layout.addWidget(add_person_widget)

        # Сохраняем ссылку для доступа
        self.add_person_widget = add_person_widget

    def switch_page(self, index):
        """Переключает активную страницу"""
        # Останавливаем камеру у предыдущей страницы
        if self.stacked_widget.currentIndex() == 0 and hasattr(self, 'recognition_widget'):
            self.recognition_widget.stop_camera()
        elif self.stacked_widget.currentIndex() == 2 and hasattr(self, 'add_person_widget'):
            self.add_person_widget.stop_camera()

        # Переключаем страницу
        self.stacked_widget.setCurrentIndex(index)

        # Обновляем состояние кнопок
        self.recognition_btn.setChecked(index == 0)
        self.database_btn.setChecked(index == 1)
        self.add_person_btn.setChecked(index == 2)

        # Обработка специальных случаев
        if index == 0:  # Распознавание
            if hasattr(self, 'recognition_widget'):
                self.recognition_widget.start_camera()
        elif index == 1:  # База данных
            if hasattr(self, 'database_widget'):
                self.database_widget.load()
        elif index == 2:  # Добавление
            if hasattr(self, 'add_person_widget'):
                self.add_person_widget.start_camera()

    def closeEvent(self, event):
        """Обработка закрытия окна"""
        # Останавливаем все камеры
        self.camera_manager.stop_camera()
        super().closeEvent(event)

class RecognitionWindow(QWidget):
    """Адаптированная версия старого окна распознавания как виджет"""

    def __init__(self, camera_manager, user_id, face_recognition_module=None):
        super().__init__()

        # Сохраняем ссылки
        self.camera_manager = camera_manager
        self.user_id = user_id
        self.timer = None

        # Модуль для распознавания лиц (должен быть передан извне)
        self.face_recognition = face_recognition_module

        self.init_ui()

    def init_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(20)

        # Заголовок
        title = QLabel("Распознавание лиц в реальном времени")
        title.setObjectName("title")
        title.setAlignment(Qt.AlignCenter)
        title.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        main_layout.addWidget(title)

        # Основной контент - используем QHBoxLayout для горизонтального расположения
        content_layout = QHBoxLayout()
        content_layout.setSpacing(30)

        # Левая часть - видео в карточке
        left_frame = QFrame()
        left_frame.setObjectName("card")
        left_frame.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding)
        left_layout = QVBoxLayout(left_frame)
        left_layout.setContentsMargins(20, 20, 20, 20)
        left_layout.setSpacing(15)

        video_title = QLabel("Видеопоток с камеры")
        video_title.setObjectName("subtitle")
        video_title.setAlignment(Qt.AlignCenter)
        left_layout.addWidget(video_title)

        # Контейнер для видео с фиксированным размером
        video_container = QFrame()
        video_container.setObjectName("videoContainer")
        video_container.setFixedSize(680, 520)  # Немного больше, чтобы было пространство
        video_container.setStyleSheet("""
            #videoContainer {
                background-color: black;
                border-radius: 12px;
                border: 3px solid #bdc3c7;
            }
        """)

        video_container_layout = QVBoxLayout(video_container)
        video_container_layout.setContentsMargins(10, 10, 10, 10)

        self.video = QLabel()
        self.video.setAlignment(Qt.AlignCenter)
        self.video.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.video.setMinimumSize(640, 480)
        self.video.setMaximumSize(640, 480)
        self.video.setText("Камера не запущена")
        self.video.setStyleSheet("""
            QLabel {
                color: white;
                font-size: 14px;
            }
        """)
        video_container_layout.addWidget(self.video, alignment=Qt.AlignCenter)

        left_layout.addWidget(video_container, alignment=Qt.AlignCenter)

        # Кнопка управления камерой
        self.camera_btn = QPushButton("▶ Запустить камеру")
        self.camera_btn.clicked.connect(self.toggle_camera)
        self.camera_btn.setMinimumHeight(40)
        left_layout.addWidget(self.camera_btn)

        content_layout.addWidget(left_frame)

        # Правая часть - информация
        right_frame = QFrame()
        right_frame.setObjectName("card")
        right_frame.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        right_layout = QVBoxLayout(right_frame)
        right_layout.setContentsMargins(20, 20, 20, 20)
        right_layout.setSpacing(15)

        info_title = QLabel("Результаты распознавания")
        info_title.setObjectName("subtitle")
        info_title.setAlignment(Qt.AlignCenter)
        right_layout.addWidget(info_title)

        # Текстовое поле с информацией
        self.info_box = QTextEdit()
        self.info_box.setReadOnly(True)
        self.info_box.setPlaceholderText("Здесь будут отображаться результаты распознавания...")
        self.info_box.setStyleSheet("""
            QTextEdit {
                border: 2px solid #bdc3c7;
                border-radius: 8px;
                padding: 15px;
                background-color: white;
                font-size: 14px;
                line-height: 1.5;
            }
            QTextEdit:focus {
                border-color: #3498db;
            }
        """)
        right_layout.addWidget(self.info_box)

        # Панель с уверенностью
        confidence_frame = QFrame()
        confidence_layout = QHBoxLayout(confidence_frame)
        confidence_layout.setContentsMargins(0, 0, 0, 0)

        confidence_label = QLabel("Уровень уверенности:")
        confidence_label.setStyleSheet("font-weight: bold; font-size: 14px;")

        self.confidence_value = QLabel("0%")
        self.confidence_value.setObjectName("confidenceValue")
        self.confidence_value.setStyleSheet("""
            #confidenceValue {
                font-size: 24px;
                font-weight: bold;
                color: #2c3e50;
                padding: 5px 15px;
                background-color: #ecf0f1;
                border-radius: 8px;
                min-width: 80px;
            }
        """)
        self.confidence_value.setAlignment(Qt.AlignCenter)

        confidence_layout.addWidget(confidence_label)
        confidence_layout.addStretch()
        confidence_layout.addWidget(self.confidence_value)

        right_layout.addWidget(confidence_frame)

        # Индикатор состояния
        self.status_label = QLabel("Статус: Ожидание запуска камеры")
        self.status_label.setStyleSheet("""
            QLabel {
                padding: 10px;
                background-color: #f8f9fa;
                border-radius: 8px;
                border: 1px solid #dee2e6;
                font-size: 13px;
                color: #6c757d;
            }
        """)
        self.status_label.setAlignment(Qt.AlignCenter)
        right_layout.addWidget(self.status_label)

        content_layout.addWidget(right_frame)

        # Добавляем основной контент
        main_layout.addLayout(content_layout)

        # Добавляем растягивающий элемент внизу
        main_layout.addStretch()

        # Устанавливаем начальный текст
        self.info_box.setText(
            "Добро пожаловать в систему распознавания лиц!\n\n"
            "Для начала работы:\n"
            "1. Нажмите кнопку 'Запустить камеру'\n"
            "2. Наведите камеру на лицо\n"
            "3. Система автоматически начнет распознавание\n\n"
            "Результаты будут отображаться здесь."
        )

    def start_camera(self):
        """Запускает камеру"""
        if not self.camera_manager.start_camera(self.user_id):
            self.status_label.setText("Статус: Ошибка запуска камеры")
            self.info_box.setText(
                "Ошибка: не удалось запустить камеру\n\nПроверьте подключение камеры и попробуйте снова.")
            return False

        # Запускаем таймер для обновления кадров
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)  # ~30 FPS

        self.camera_btn.setText("⏸ Остановить камеру")
        self.status_label.setText("Статус: Камера активна - наведите на лицо")
        return True

    def stop_camera(self):
        """Останавливает камеру"""
        if self.timer:
            self.timer.stop()
            self.timer = None

        self.camera_manager.stop_camera(self.user_id)
        self.camera_btn.setText("▶ Запустить камеру")

        # Очищаем видео и показываем черный фон
        self.video.clear()
        self.video.setText("Камера остановлена")
        self.video.setStyleSheet("""
            QLabel {
                background-color: black;
                color: white;
                font-size: 14px;
                border-radius: 8px;
            }
        """)

        self.status_label.setText("Статус: Камера остановлена")
        self.confidence_value.setText("0%")

        self.info_box.setText(
            "Камера остановлена\n\n"
            "Для возобновления работы нажмите кнопку 'Запустить камеру'.\n\n"
            "Ранее распознанные лица:\n"
            "• Результаты будут отображаться здесь после запуска камеры."
        )

    def toggle_camera(self):
        """Переключает состояние камеры"""
        if self.timer and self.timer.isActive():
            self.stop_camera()
        else:
            self.start_camera()

    def update_frame(self):
        """Обновляет кадр с камеры и выполняет распознавание"""
        # Проверяем, доступна ли камера
        if not self.camera_manager.is_camera_available(self.user_id):
            return

        # Получаем кадр
        frame = self.camera_manager.get_frame()
        if frame is None:
            return

        # Инициализируем переменные для распознавания
        best_score = 0
        best_person_id = None
        recognized_person = None

        # Обрабатываем кадр, если доступен модуль распознавания
        if self.face_recognition:
            # Извлекаем лицо из кадра
            face = self.face_recognition.extract_face(frame)

            if face is not None:
                # Получаем все фото из базы данных
                all_photos = self.face_recognition.get_all_photos()

                for person_id, blob in all_photos:
                    # Декодируем изображение из базы данных
                    db_img = cv2.imdecode(
                        np.frombuffer(blob, np.uint8),
                        cv2.IMREAD_GRAYSCALE
                    )

                    # Сравниваем лица
                    score = self.face_recognition.compare_faces(face, db_img)

                    if score > best_score:
                        best_score = score
                        best_person_id = person_id

                # Получаем информацию о распознанном человеке
                if best_score >= 50 and best_person_id is not None:
                    recognized_person = self.face_recognition.get_person_by_id(best_person_id)

        # Обновление информации
        if recognized_person is not None and best_score >= 50:
            self.update_person_info(recognized_person, best_score)
        else:
            self.update_person_info(None, best_score)

        # Отображение видео с правильным масштабированием
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape

        # Создаем QImage
        img = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)

        # Масштабируем изображение под размер контейнера
        pixmap = QPixmap.fromImage(img)
        scaled_pixmap = pixmap.scaled(
            self.video.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )

        # Устанавливаем пиксмап
        self.video.setPixmap(scaled_pixmap)
        self.video.setAlignment(Qt.AlignCenter)

    def update_person_info(self, person, similarity):
        """Обновляет информацию о распознанном человеке"""
        self.confidence_value.setText(f"{similarity:.1f}%")

        # Обновляем цвет уверенности
        if similarity >= 80:
            color = "#27ae60"  # зеленый
        elif similarity >= 60:
            color = "#f39c12"  # оранжевый
        else:
            color = "#e74c3c"  # красный

        self.confidence_value.setStyleSheet(f"""
            #confidenceValue {{
                font-size: 24px;
                font-weight: bold;
                color: white;
                padding: 5px 15px;
                background-color: {color};
                border-radius: 8px;
                min-width: 80px;
            }}
        """)

        if person is None:
            self.status_label.setText("Статус: Неизвестное лицо")
            self.info_box.setText(
                "Результат распознавания:\n\n"
                "⚠️ Неопознанное лицо\n\n"
                "Человек не найден в базе данных.\n"
                f"Совпадение: {similarity:.1f}%\n\n"
                "Рекомендации:\n"
                "• Убедитесь, что лицо хорошо освещено\n"
                "• Лицо должно быть полностью видно в кадре\n"
                "• Попробуйте добавить человека в базу данных"
            )
        else:
            self.status_label.setText(f"Статус: Распознан - {person[1]} {person[2]}")
            self.info_box.setText(
                "✅ Лицо распознано!\n\n"
                f"👤 Имя: {person[1]}\n"
                f"👥 Фамилия: {person[2]}\n"
                f"🎓 Группа: {person[3]}\n"
                f"📝 Описание: {person[4] or 'не указано'}\n\n"
                f"🎯 Уверенность: {similarity:.1f}%\n\n"
                f"🔢 ID в системе: {person[0]}"
            )

    def set_face_recognition_module(self, face_recognition_module):
        """Устанавливает модуль для распознавания лиц"""
        self.face_recognition = face_recognition_module

class DatabaseWindow(QWidget):
    """Адаптированная версия старого окна базы данных как виджет"""

    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(20)

        # Заголовок
        title = QLabel("База данных лиц")
        title.setObjectName("title")
        title.setAlignment(Qt.AlignCenter)
        title.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        main_layout.addWidget(title)

        # Основной контент - используем QHBoxLayout
        content_layout = QHBoxLayout()
        content_layout.setSpacing(30)
        content_layout.setContentsMargins(0, 0, 0, 0)

        # Левая часть - таблица с кнопками управления
        left_frame = QFrame()
        left_frame.setObjectName("card")
        left_frame.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        left_layout = QVBoxLayout(left_frame)
        left_layout.setContentsMargins(20, 20, 20, 20)
        left_layout.setSpacing(15)

        table_title = QLabel("Зарегистрированные пользователи")
        table_title.setObjectName("subtitle")
        table_title.setAlignment(Qt.AlignCenter)
        left_layout.addWidget(table_title)

        # Создаем таблицу
        self.table = QTableView()
        self.model = QStandardItemModel(0, 4)
        self.model.setHorizontalHeaderLabels(["ID", "Имя", "Фамилия", "Группа"])

        # Настройки таблицы
        self.table.setModel(self.model)
        self.table.setSelectionBehavior(QTableView.SelectRows)
        self.table.setSelectionMode(QTableView.SingleSelection)
        self.table.setEditTriggers(QTableView.NoEditTriggers)
        self.table.clicked.connect(self.show_photos)

        # Настройка размеров колонок
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.verticalHeader().setVisible(False)

        # Стиль таблицы
        self.table.setStyleSheet("""
            QTableView {
                border: 1px solid #bdc3c7;
                border-radius: 8px;
                background-color: white;
                alternate-background-color: #f8f9fa;
                gridline-color: #e9ecef;
            }
            QTableView::item {
                padding: 10px;
                border-bottom: 1px solid #e9ecef;
            }
            QTableView::item:selected {
                background-color: #3498db;
                color: white;
            }
            QHeaderView::section {
                background-color: #2c3e50;
                color: white;
                padding: 12px;
                border: none;
                font-weight: bold;
            }
        """)

        # Устанавливаем фиксированную высоту для таблицы
        self.table.setMinimumHeight(400)
        self.table.setMaximumHeight(500)

        left_layout.addWidget(self.table)

        # Панель кнопок управления - отдельный фрейм
        buttons_frame = QFrame()
        buttons_layout = QHBoxLayout(buttons_frame)
        buttons_layout.setContentsMargins(0, 10, 0, 0)
        buttons_layout.setSpacing(15)

        btn_refresh = QPushButton("🔄 Обновить")
        btn_refresh.setMinimumHeight(40)
        btn_refresh.clicked.connect(self.load)

        self.btn_delete = QPushButton("🗑️ Удалить выбранного")
        self.btn_delete.setMinimumHeight(40)
        self.btn_delete.clicked.connect(self.remove)
        self.btn_delete.setEnabled(False)

        buttons_layout.addWidget(btn_refresh)
        buttons_layout.addStretch()
        buttons_layout.addWidget(self.btn_delete)

        left_layout.addWidget(buttons_frame)

        content_layout.addWidget(left_frame)

        # Правая часть - фотография
        right_frame = QFrame()
        right_frame.setObjectName("card")
        right_frame.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        right_layout = QVBoxLayout(right_frame)
        right_layout.setContentsMargins(20, 20, 20, 20)
        right_layout.setSpacing(15)

        photo_title = QLabel("Фотографии пользователя")
        photo_title.setObjectName("subtitle")
        photo_title.setAlignment(Qt.AlignCenter)
        right_layout.addWidget(photo_title)

        # Контейнер для фотографии с фиксированными размерами
        photo_container_frame = QFrame()
        photo_container_frame.setObjectName("photoContainer")
        photo_container_frame.setMinimumHeight(350)
        photo_container_frame.setMaximumHeight(400)
        photo_container_frame.setStyleSheet("""
            #photoContainer {
                background-color: #f8f9fa;
                border: 2px dashed #bdc3c7;
                border-radius: 12px;
            }
        """)

        photo_container_layout = QVBoxLayout(photo_container_frame)
        photo_container_layout.setContentsMargins(20, 20, 20, 20)

        self.photo_container = QLabel("Выберите пользователя из таблицы")
        self.photo_container.setAlignment(Qt.AlignCenter)
        self.photo_container.setWordWrap(True)
        self.photo_container.setStyleSheet("""
            QLabel {
                color: #6c757d;
                font-size: 14px;
                padding: 20px;
            }
        """)

        photo_container_layout.addWidget(self.photo_container)
        right_layout.addWidget(photo_container_frame)

        # Информация о выбранном пользователе
        self.user_info_label = QLabel("Информация не выбрана")
        self.user_info_label.setAlignment(Qt.AlignCenter)
        self.user_info_label.setWordWrap(True)
        self.user_info_label.setStyleSheet("""
            QLabel {
                padding: 15px;
                background-color: #e9ecef;
                border-radius: 8px;
                font-size: 13px;
                color: #495057;
            }
        """)
        right_layout.addWidget(self.user_info_label)

        content_layout.addWidget(right_frame)

        # Добавляем основной контент
        main_layout.addLayout(content_layout)

        # Статус бар внизу
        self.status_label = QLabel()
        self.status_label.setObjectName("statusLabel")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setStyleSheet("""
            #statusLabel {
                padding: 10px;
                background-color: #2c3e50;
                color: white;
                border-radius: 8px;
                font-size: 13px;
            }
        """)
        main_layout.addWidget(self.status_label)

        # Загрузка данных
        self.load()

    def load(self):
        """Загружает данные в таблицу"""
        try:
            # Удаляем старые данные
            self.model.removeRows(0, self.model.rowCount())

            # Получаем данные
            persons = get_all_persons()

            # Заполняем таблицу
            for p in persons:
                self.model.appendRow([
                    QStandardItem(str(p[0])),
                    QStandardItem(p[1]),
                    QStandardItem(p[2]),
                    QStandardItem(p[3])
                ])

            # Обновляем статус
            self.status_label.setText(
                f"✓ Загружено записей: {len(persons)} | Последнее обновление: {self.get_current_time()}")

        except Exception as e:
            self.status_label.setText(f"✗ Ошибка загрузки: {str(e)}")

    def get_current_time(self):
        """Возвращает текущее время в формате HH:MM:SS"""
        from datetime import datetime
        return datetime.now().strftime("%H:%M:%S")

    def current_person_id(self):
        idx = self.table.currentIndex()
        if not idx.isValid():
            return None

        # Получаем ID из первого столбца выбранной строки
        item = self.model.item(idx.row(), 0)
        if item:
            try:
                return int(item.text())
            except:
                return None
        return None

    def remove(self):
        pid = self.current_person_id()
        if pid is None:
            return

        reply = QMessageBox.question(
            self, 'Подтверждение удаления',
            f'Вы уверены, что хотите удалить пользователя с ID {pid}?\n\n'
            'Внимание: все связанные фотографии также будут удалены без возможности восстановления.',
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No
        )

        if reply == QMessageBox.Yes:
            try:
                delete_person(pid)
                self.load()
                self.photo_container.setText("Пользователь удален")
                self.photo_container.setPixmap(QPixmap())
                self.btn_delete.setEnabled(False)
                self.user_info_label.setText("Информация не выбрана")
                self.status_label.setText(f"✓ Пользователь ID {pid} удален | {self.get_current_time()}")
            except Exception as e:
                QMessageBox.critical(self, "Ошибка", f"Не удалось удалить пользователя: {str(e)}")

    def show_photos(self):
        """Показывает фотографии выбранного пользователя"""
        pid = self.current_person_id()
        if pid is None:
            self.btn_delete.setEnabled(False)
            return

        self.btn_delete.setEnabled(True)

        try:
            photos = get_photos_by_person(pid)

            if not photos:
                self.photo_container.setText("Нет фотографий")
                self.photo_container.setPixmap(QPixmap())
                # Получаем информацию о пользователе
                from database import get_person_by_id
                person = get_person_by_id(pid)
                if person:
                    self.user_info_label.setText(
                        f"👤 {person[1]} {person[2]}\n"
                        f"🎓 Группа: {person[3]}\n"
                        f"📝 Описание: {person[4] or 'не указано'}\n"
                        f"🔢 ID: {pid}"
                    )
                return

            # Показываем первую фотографию
            blob = photos[0][1]
            img = cv2.imdecode(np.frombuffer(blob, np.uint8), cv2.IMREAD_COLOR)

            if img is not None:
                rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb.shape
                qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
                pixmap = QPixmap.fromImage(qimg)

                # Масштабируем по размеру контейнера
                scaled_pixmap = pixmap.scaled(
                    self.photo_container.size().width() - 40,
                    self.photo_container.size().height() - 40,
                    Qt.KeepAspectRatio,
                    Qt.SmoothTransformation
                )
                self.photo_container.setPixmap(scaled_pixmap)
                self.photo_container.setText("")

                # Получаем информацию о пользователе
                from database import get_person_by_id
                person = get_person_by_id(pid)
                if person:
                    self.user_info_label.setText(
                        f"👤 {person[1]} {person[2]}\n"
                        f"🎓 Группа: {person[3]}\n"
                        f"📝 Описание: {person[4] or 'не указано'}\n"
                        f"🔢 ID: {pid}\n"
                        f"📸 Фотографий: {len(photos)}"
                    )
            else:
                self.photo_container.setText("Ошибка загрузки изображения")
                self.photo_container.setPixmap(QPixmap())
        except Exception as e:
            self.photo_container.setText(f"Ошибка: {str(e)}")


class AddPersonWindow(QWidget):
    """Виджет добавления нового человека"""

    def __init__(self, camera_manager, user_id):
        super().__init__()

        # Сохраняем ссылки
        self.camera_manager = camera_manager
        self.user_id = user_id

        # Импортируем здесь, чтобы избежать циклических импортов
        from PyQt5.QtWidgets import QLineEdit, QFileDialog, QProgressBar, QStackedWidget
        from person_service import add_person as add_person_service
        from image_utils import image_to_bytes
        from database import add_photo

        self.QLineEdit = QLineEdit
        self.QFileDialog = QFileDialog
        self.QProgressBar = QProgressBar
        self.QStackedWidget = QStackedWidget
        self.add_person_service = add_person_service
        self.image_to_bytes = image_to_bytes
        self.add_photo = add_photo

        # Инициализируем переменные
        self.person_created = False
        self.person_id = None
        self.capture_timer = None
        self.preview_timer = None
        self.photos_captured = 0
        self.total_photos_to_capture = 200
        self.is_capturing = False

        self.init_ui()

    def init_ui(self):
        # Основной layout
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(20)

        # Создаем StackedWidget для переключения между режимами
        self.stacked_widget = self.QStackedWidget()

        # Создаем режимы
        self.create_input_mode()
        self.create_camera_mode()

        # Добавляем оба режима в StackedWidget
        self.stacked_widget.addWidget(self.input_widget)
        self.stacked_widget.addWidget(self.camera_widget)

        # Показываем режим ввода данных
        self.stacked_widget.setCurrentWidget(self.input_widget)

        # Добавляем StackedWidget в основной layout
        main_layout.addWidget(self.stacked_widget)

    def create_input_mode(self):
        """Создает виджет для режима ввода данных"""
        self.input_widget = QWidget()
        layout = QVBoxLayout(self.input_widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(20)

        # Заголовок
        title = QLabel("Добавление нового человека")
        title.setObjectName("title")
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)

        # Форма ввода данных
        form_frame = QFrame()
        form_frame.setObjectName("card")
        form_layout = QVBoxLayout(form_frame)
        form_layout.setContentsMargins(30, 25, 30, 25)
        form_layout.setSpacing(15)

        form_title = QLabel("📝 Основная информация")
        form_title.setObjectName("subtitle")
        form_title.setAlignment(Qt.AlignCenter)
        form_layout.addWidget(form_title)

        # Поля ввода
        self.first = self.QLineEdit()
        self.first.setPlaceholderText("Имя *")
        self.first.setMinimumHeight(40)

        self.last = self.QLineEdit()
        self.last.setPlaceholderText("Фамилия *")
        self.last.setMinimumHeight(40)

        self.group = self.QLineEdit()
        self.group.setPlaceholderText("Группа *")
        self.group.setMinimumHeight(40)

        self.desc = self.QLineEdit()
        self.desc.setPlaceholderText("Описание (необязательно)")
        self.desc.setMinimumHeight(40)

        form_layout.addWidget(self.first)
        form_layout.addWidget(self.last)
        form_layout.addWidget(self.group)
        form_layout.addWidget(self.desc)
        layout.addWidget(form_frame)

        # Кнопки
        buttons_frame = QFrame()
        buttons_layout = QVBoxLayout(buttons_frame)
        buttons_layout.setSpacing(15)

        # Кнопка загрузки из файлов
        self.btn_files = QPushButton("📁 Добавить фото из файлов")
        self.btn_files.setMinimumHeight(45)
        self.btn_files.clicked.connect(self.from_files)

        # Кнопка съемки с камеры
        self.btn_start_camera = QPushButton("📸 Добавить фото с камеры")
        self.btn_start_camera.setMinimumHeight(45)
        self.btn_start_camera.clicked.connect(self.start_camera_mode)

        buttons_layout.addWidget(self.btn_files)
        buttons_layout.addWidget(self.btn_start_camera)
        layout.addWidget(buttons_frame)

        # Информационное поле
        self.info_label = QLabel("Заполните поля и выберите способ добавления фото")
        self.info_label.setAlignment(Qt.AlignCenter)
        self.info_label.setWordWrap(True)
        layout.addWidget(self.info_label)

        # Растягивающий элемент
        layout.addStretch()

    def create_camera_mode(self):
        """Создает виджет для режима съемки камерой"""
        self.camera_widget = QWidget()
        layout = QVBoxLayout(self.camera_widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)  # Уменьшаем spacing

        # Заголовок
        title = QLabel("Съемка с камеры")
        title.setObjectName("title")
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)

        # Видео с камеры - ФИКСИРОВАННЫЙ размер
        self.camera_display = QLabel("Запуск камеры...")
        self.camera_display.setAlignment(Qt.AlignCenter)
        self.camera_display.setFixedSize(640, 480)  # ФИКСИРОВАННЫЙ размер
        self.camera_display.setStyleSheet("""
            QLabel {
                background-color: black;
                border-radius: 8px;
                border: 2px solid #bdc3c7;
                color: white;
                font-size: 14px;
            }
        """)
        layout.addWidget(self.camera_display, alignment=Qt.AlignCenter)

        # Прогресс бар
        self.progress_bar = self.QProgressBar()
        self.progress_bar.setRange(0, self.total_photos_to_capture)
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("Снято фото: %v/%m")
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setMinimumHeight(30)
        layout.addWidget(self.progress_bar)

        # Кнопка остановки
        self.btn_stop_camera = QPushButton("⏹️ Остановить запись и вернуться")
        self.btn_stop_camera.setMinimumHeight(45)
        self.btn_stop_camera.setStyleSheet("""
            QPushButton {
                background-color: #e74c3c;
                font-size: 14px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #c0392b;
            }
        """)
        self.btn_stop_camera.clicked.connect(self.stop_camera_mode)
        layout.addWidget(self.btn_stop_camera)

        # Растягивающий элемент
        layout.addStretch()

    def start_camera_mode(self):
        """Переключает в режим камеры"""
        # Проверяем обязательные поля
        if not self.validate_fields():
            return

        # Создаем человека если еще не создан
        self.create_person_once()

        # Запускаем камеру
        if not self.camera_manager.start_camera(self.user_id):
            self.info_label.setText("⚠️ Не удалось запустить камеру")
            return

        # Переключаемся в режим камеры
        self.stacked_widget.setCurrentWidget(self.camera_widget)

        # Запускаем предпросмотр и съемку
        self.start_preview()
        self.start_capture()

    def stop_camera_mode(self):
        """Выходит из режима камеры"""
        self.stop_capture()
        self.camera_manager.stop_camera(self.user_id)

        # Переключаемся обратно в режим ввода
        self.stacked_widget.setCurrentWidget(self.input_widget)

        if self.photos_captured > 0:
            QMessageBox.information(
                self,
                "Съемка завершена",
                f"✅ Съемка завершена!\n\nСохранено фото: {self.photos_captured}"
            )

    def start_preview(self):
        """Запускает предпросмотр камеры"""
        if self.preview_timer:
            self.preview_timer.stop()

        self.preview_timer = QTimer()
        self.preview_timer.timeout.connect(self.update_preview)
        self.preview_timer.start(30)  # ~30 FPS

    def start_capture(self):
        """Запускает автоматическую съемку"""
        self.photos_captured = 0
        self.is_capturing = True
        self.progress_bar.setValue(0)

        # ЗАПУСКАЕМ ТАЙМЕР ДЛЯ СЪЕМКИ
        if self.capture_timer:
            self.capture_timer.stop()

        self.capture_timer = QTimer()
        self.capture_timer.timeout.connect(self.capture_single_photo)
        self.capture_timer.start(100)  # 1 фото каждые 100мс

    def update_preview(self):
        """Обновляет изображение с камеры - ФИКСИРОВАННЫЙ размер"""
        if not self.camera_manager.is_camera_available(self.user_id):
            return

        frame = self.camera_manager.get_frame()
        if frame is None:
            return

        # Фиксируем размер для отображения
        target_width = 640
        target_height = 480

        # Конвертируем кадр для отображения
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Изменяем размер изображения до фиксированного
        rgb_resized = cv2.resize(rgb, (target_width, target_height))

        h, w, ch = rgb_resized.shape

        # Создаем QImage с фиксированным размером
        img = QImage(rgb_resized.data, w, h, ch * w, QImage.Format_RGB888)

        # Создаем пиксмап фиксированного размера
        pixmap = QPixmap.fromImage(img)

        # Устанавливаем пиксмап
        self.camera_display.setPixmap(pixmap)

    def capture_single_photo(self):
        """Захватывает одно фото"""
        if self.photos_captured >= self.total_photos_to_capture:
            self.stop_camera_mode()
            return

        # Получаем кадр с камеры
        frame = self.camera_manager.get_frame()
        if frame is None:
            return

        # Извлекаем лицо
        face = extract_face(frame)
        if face is not None:
            # Сохраняем фото
            data = self.image_to_bytes(face)
            self.add_photo(self.person_id, f"auto_capture_{self.photos_captured}", "jpg", len(data), data)

            self.photos_captured += 1
            self.progress_bar.setValue(self.photos_captured)

        # Если достигли лимита - останавливаем
        if self.photos_captured >= self.total_photos_to_capture:
            self.stop_camera_mode()

    def stop_capture(self):
        """Останавливает съемку"""
        self.is_capturing = False

        if self.capture_timer:
            self.capture_timer.stop()
            self.capture_timer = None

        if self.preview_timer:
            self.preview_timer.stop()
            self.preview_timer = None

    def from_files(self):
        """Добавление фото из файлов"""
        if not self.validate_fields():
            return

        pid = self.create_person_once()
        files, _ = self.QFileDialog.getOpenFileNames(
            self,
            "Выбор изображений",
            "",
            "Изображения (*.jpg *.jpeg *.png *.bmp *.gif)"
        )

        if not files:
            return

        count = 0
        for f in files:
            img = cv2.imread(f)
            if img is None:
                continue

            face = extract_face(img)
            if face is None:
                continue

            data = self.image_to_bytes(face)
            self.add_photo(pid, f, "jpg", len(data), data)
            count += 1

        QMessageBox.information(
            self,
            "Успешно!",
            f"✅ Изображения успешно добавлены!\n\nЗагружено: {count} фото"
        )

        self.info_label.setText(f"Последнее действие: загружено {count} фото")

    def create_person_once(self):
        """Создаёт человека только один раз"""
        if not self.person_created:
            self.person_id = self.add_person_service(
                self.first.text(),
                self.last.text(),
                self.group.text(),
                self.desc.text() or None
            )
            self.person_created = True
            self.info_label.setText(f"Создан профиль ID: {self.person_id}")
        return self.person_id

    def validate_fields(self):
        if not self.first.text() or not self.last.text() or not self.group.text():
            QMessageBox.warning(
                self,
                "Заполните обязательные поля",
                "Пожалуйста, заполните все поля, отмеченные звездочкой (*):\n\n• Имя\n• Фамилия\n• Группа"
            )
            return False
        return True

    def start_camera(self):
        """Метод для MainWindow - запускает камеру при переходе на эту страницу"""
        pass  # Камера запускается только в режиме съемки

    def stop_camera(self):
        """Метод для MainWindow - останавливает камеру при уходе с этой страницы"""
        self.stop_capture()
        self.camera_manager.stop_camera(self.user_id)