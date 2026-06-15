DARK_STYLE = """
QWidget {
    background-color: #0b1220;
    color: #e5e7eb;
    font-family: Segoe UI;
    font-size: 14px;
}

QLabel {
    color: #e5e7eb;
}

#title {
    font-size: 22px;
    font-weight: bold;
    color: #60a5fa;
    padding: 10px;
}

QPushButton {
    background-color: #2563eb;
    color: white;
    padding: 10px;
    border-radius: 10px;
    font-weight: bold;
}

QPushButton:hover {
    background-color: #1d4ed8;
}

QPushButton:pressed {
    background-color: #1e40af;
}

QPushButton:disabled {
    background-color: #374151;
    color: #9ca3af;
}

QPushButton#cancelButton {
    background-color: #dc2626;
    color: white;
}

QPushButton#cancelButton:hover {
    background-color: #b91c1c;
}

QPushButton#themeButton {
    background-color: #111827;
    color: #e5e7eb;
    border: 1px solid #374151;
    padding: 8px 12px;
    border-radius: 8px;
}

QPushButton#themeButton:hover {
    background-color: #1f2937;
}

QComboBox {
    background-color: #111827;
    padding: 6px;
    border-radius: 6px;
    color: white;
    border: 1px solid #374151;
}

QProgressBar {
    border: 1px solid #374151;
    border-radius: 6px;
    text-align: center;
    background: #111827;
}

QProgressBar::chunk {
    background-color: #2563eb;
    border-radius: 6px;
}

QTextEdit {
    background-color: #0f172a;
    border: 1px solid #374151;
    border-radius: 8px;
    padding: 8px;
    color: #e5e7eb;
}
"""

LIGHT_STYLE = """
QWidget {
    background-color: #f8fafc;
    color: #111827;
    font-family: Segoe UI;
    font-size: 14px;
}

QLabel {
    color: #111827;
}

#title {
    font-size: 22px;
    font-weight: bold;
    color: #1d4ed8;
    padding: 10px;
}

QPushButton {
    background-color: #2563eb;
    color: white;
    padding: 10px;
    border-radius: 10px;
    font-weight: bold;
}

QPushButton:hover {
    background-color: #1d4ed8;
}

QPushButton:pressed {
    background-color: #1e40af;
}

QPushButton:disabled {
    background-color: #d1d5db;
    color: #6b7280;
}

QPushButton#cancelButton {
    background-color: #dc2626;
    color: white;
}

QPushButton#cancelButton:hover {
    background-color: #b91c1c;
}

QPushButton#themeButton {
    background-color: #ffffff;
    color: #111827;
    border: 1px solid #cbd5e1;
    padding: 8px 12px;
    border-radius: 8px;
}

QPushButton#themeButton:hover {
    background-color: #e5e7eb;
}

QComboBox {
    background-color: #ffffff;
    padding: 6px;
    border-radius: 6px;
    color: #111827;
    border: 1px solid #cbd5e1;
}

QProgressBar {
    border: 1px solid #cbd5e1;
    border-radius: 6px;
    text-align: center;
    background: #ffffff;
    color: #111827;
}

QProgressBar::chunk {
    background-color: #2563eb;
    border-radius: 6px;
}

QTextEdit {
    background-color: #ffffff;
    border: 1px solid #cbd5e1;
    border-radius: 8px;
    padding: 8px;
    color: #111827;
}
"""

STYLE = DARK_STYLE
