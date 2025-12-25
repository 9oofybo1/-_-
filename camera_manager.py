import cv2
import threading
from PyQt5.QtCore import QObject, pyqtSignal, QTimer


class CameraManager(QObject):
    """Глобальный менеджер камеры для безопасного доступа из разных виджетов"""

    frame_ready = pyqtSignal(object)  # Сигнал с новым кадром

    def __init__(self):
        super().__init__()
        self.cap = None
        self.timer = None
        self.is_running = False
        self.current_consumer = None  # Текущий виджет, использующий камеру
        self.lock = threading.Lock()

    def start_for_widget(self, widget_id):
        """Запуск камеры для конкретного виджета"""
        with self.lock:
            if self.current_consumer and self.current_consumer != widget_id:
                self.stop_camera()

            self.current_consumer = widget_id

            if not self.is_running:
                self.start_camera()

    def stop_for_widget(self, widget_id):
        """Остановка камеры для виджета"""
        with self.lock:
            if self.current_consumer == widget_id:
                self.stop_camera()
                self.current_consumer = None

    def start_camera(self):
        """Запуск камеры"""
        if self.is_running:
            return

        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            print("Ошибка: не удалось открыть камеру")
            return

        self.is_running = True
        self.timer = QTimer()
        self.timer.timeout.connect(self.capture_frame)
        self.timer.start(30)  # ~30 FPS

    def stop_camera(self):
        """Остановка камеры"""
        if not self.is_running:
            return

        if self.timer:
            self.timer.stop()
            self.timer = None

        if self.cap and self.cap.isOpened():
            self.cap.release()
            self.cap = None

        self.is_running = False

    def capture_frame(self):
        """Захват кадра и отправка сигнала"""
        if self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                self.frame_ready.emit(frame)

    def get_single_frame(self):
        """Получить один кадр (для фотографий)"""
        if self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            return frame if ret else None
        return None


# Глобальный экземпляр менеджера
camera_manager = CameraManager()