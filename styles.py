"""
Стили для приложения Facer
"""

# Цветовая палитра
COLORS = {
    'primary': '#2c3e50',
    'primary_dark': '#1a252f',
    'primary_light': '#34495e',
    'accent': '#3498db',
    'accent_hover': '#2980b9',
    'success': '#27ae60',
    'warning': '#f39c12',
    'danger': '#e74c3c',
    'background': '#ecf0f1',
    'card_bg': '#ffffff',
    'text': '#2c3e50',
    'text_light': '#7f8c8d',
    'border': '#bdc3c7',
    'hover': '#f5f7fa'
}

# Основные стили приложения
STYLE = f"""
/* Основные настройки */
QMainWindow, QWidget {{
    background-color: {COLORS['background']};
    font-family: 'Segoe UI', 'Arial', sans-serif;
    color: {COLORS['text']};
}}

/* Заголовки */
QLabel#title {{
    font-size: 28px;
    font-weight: bold;
    color: {COLORS['primary']};
    padding: 10px;
    margin-bottom: 20px;
}}

QLabel#subtitle {{
    font-size: 18px;
    font-weight: 600;
    color: {COLORS['primary_light']};
    margin: 10px 0;
}}

/* Панель навигации */
QFrame#navFrame {{
    background-color: {COLORS['primary_dark']};
    border-right: 2px solid {COLORS['border']};
}}

QPushButton#navButton {{
    background-color: transparent;
    color: {COLORS['text_light']};
    border: none;
    border-radius: 8px;
    text-align: left;
    padding: 15px 20px;
    font-size: 14px;
    font-weight: 500;
    margin: 2px 10px;
}}

QPushButton#navButton:hover {{
    background-color: {COLORS['primary_light']};
    color: white;
}}

QPushButton#navButton:checked {{
    background-color: {COLORS['accent']};
    color: white;
    font-weight: bold;
}}

/* Карточки и контейнеры */
QFrame#card {{
    background-color: {COLORS['card_bg']};
    border-radius: 12px;
    border: 1px solid {COLORS['border']};
    padding: 20px;
}}

/* Кнопки */
QPushButton {{
    background-color: {COLORS['accent']};
    color: white;
    border: none;
    border-radius: 8px;
    padding: 12px 24px;
    font-size: 14px;
    font-weight: bold;
    min-height: 40px;
}}

QPushButton:hover {{
    background-color: {COLORS['accent_hover']};
}}

QPushButton:pressed {{
    background-color: {COLORS['primary']};
}}

QPushButton:disabled {{
    background-color: {COLORS['border']};
    color: {COLORS['text_light']};
}}

/* Поля ввода */
QLineEdit {{
    border: 2px solid {COLORS['border']};
    border-radius: 6px;
    padding: 12px;
    font-size: 14px;
    background-color: white;
    selection-background-color: {COLORS['accent']};
}}

QLineEdit:focus {{
    border-color: {COLORS['accent']};
}}

QLineEdit:disabled {{
    background-color: {COLORS['hover']};
    color: {COLORS['text_light']};
}}

/* Выпадающие списки */
QComboBox {{
    border: 2px solid {COLORS['border']};
    border-radius: 6px;
    padding: 10px;
    background-color: white;
    min-height: 40px;
}}

QComboBox:focus {{
    border-color: {COLORS['accent']};
}}

QComboBox::drop-down {{
    border: none;
}}

QComboBox QAbstractItemView {{
    border: 1px solid {COLORS['border']};
    background-color: white;
    selection-background-color: {COLORS['accent']};
}}

/* Таблицы */
QTableView {{
    background-color: white;
    border: 1px solid {COLORS['border']};
    border-radius: 8px;
    gridline-color: {COLORS['border']};
    alternate-background-color: {COLORS['hover']};
}}

QTableView::item {{
    padding: 10px;
    border-bottom: 1px solid {COLORS['border']};
}}

QTableView::item:selected {{
    background-color: {COLORS['accent']};
    color: white;
}}

QHeaderView::section {{
    background-color: {COLORS['primary']};
    color: white;
    padding: 12px;
    border: none;
    font-weight: bold;
}}

/* Прогресс-бары */
QProgressBar {{
    border: 1px solid {COLORS['border']};
    border-radius: 4px;
    background-color: white;
    text-align: center;
}}

QProgressBar::chunk {{
    background-color: {COLORS['accent']};
    border-radius: 4px;
}}

/* Текстовые поля */
QTextEdit {{
    border: 2px solid {COLORS['border']};
    border-radius: 6px;
    padding: 10px;
    background-color: white;
    font-size: 14px;
}}

QTextEdit:focus {{
    border-color: {COLORS['accent']};
}}

/* Группы и разделители */
QGroupBox {{
    font-weight: bold;
    border: 2px solid {COLORS['border']};
    border-radius: 8px;
    margin-top: 20px;
    padding-top: 10px;
}}

QGroupBox::title {{
    subcontrol-origin: margin;
    left: 10px;
    padding: 0 10px;
}}

/* Полосы прокрутки */
QScrollBar:vertical {{
    border: none;
    background-color: {COLORS['hover']};
    width: 12px;
    border-radius: 6px;
}}

QScrollBar::handle:vertical {{
    background-color: {COLORS['primary_light']};
    border-radius: 6px;
    min-height: 30px;
}}

QScrollBar::handle:vertical:hover {{
    background-color: {COLORS['accent']};
}}

QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
    border: none;
    background: none;
}}

/* Индикаторы состояния */
QLabel#success {{
    color: {COLORS['success']};
    font-weight: bold;
}}

QLabel#warning {{
    color: {COLORS['warning']};
    font-weight: bold;
}}

QLabel#error {{
    color: {COLORS['danger']};
    font-weight: bold;
}}

/* Видео-контейнер */
QLabel#videoLabel {{
    background-color: black;
    border-radius: 8px;
    border: 2px solid {COLORS['border']};
}}

/* Иконки в кнопках */
QPushButton[icon] {{
    text-align: left;
    padding-left: 50px;
}}

/* Специальные стили для страниц */
QWidget#recognitionPage {{
    background-color: {COLORS['background']};
}}

QWidget#databasePage {{
    background-color: {COLORS['background']};
}}

QWidget#addPersonPage {{
    background-color: {COLORS['background']};
}}
"""