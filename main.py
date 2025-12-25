import sys
from PyQt5.QtWidgets import QApplication
from main_window import MainWindow
from database import init_db


def main():
    # Инициализация базы данных
    init_db()

    # Создание приложения
    app = QApplication(sys.argv)

    # Создание главного окна
    window = MainWindow()
    window.show()

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()