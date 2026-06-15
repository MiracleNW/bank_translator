import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

try:
    from src.torch_runtime import import_torch_early
    import_torch_early()
except Exception as exc:
    message = (
        "Failed to preload PyTorch before starting the GUI.\n\n"
        "Run build_cuda_exe.bat again.\n\n"
        f"Original error: {exc}"
    )
    if sys.platform.startswith("win"):
        try:
            import ctypes
            ctypes.windll.user32.MessageBoxW(None, message, "PyTorch startup error", 0x10)
        except Exception:
            pass
    raise RuntimeError(message) from exc

from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QLabel, QFileDialog, QComboBox, QTextEdit, QProgressBar
)

from gui_core.controller import AppController
from gui_core.state import AppState
from app.style import DARK_STYLE, LIGHT_STYLE

# Main window
class TranslatorApp(QWidget):

    def __init__(self):
        super().__init__()

        self.setWindowTitle("Bank Translation")
        self.setGeometry(300, 200, 800, 540)

        self.file_path = None
        self.controller = None
        self.theme_name = "dark"

        self.init_ui()
        self.apply_style()

        self.controller = AppController(self)
        self.controller.start_model_preload()

    # UI
    def init_ui(self):

        layout = QVBoxLayout()

        header = QHBoxLayout()
        self.title = QLabel("Bank Translation")
        self.title.setObjectName("title")
        header.addWidget(self.title, 1)

        self.btn_theme = QPushButton("Светлая тема")
        self.btn_theme.setObjectName("themeButton")
        self.btn_theme.clicked.connect(self.toggle_theme)
        header.addWidget(self.btn_theme, 0)
        layout.addLayout(header)

        self.file_label = QLabel("Файл не выбран")
        layout.addWidget(self.file_label)

        self.btn_select = QPushButton("Выбрать DOCX")
        self.btn_select.clicked.connect(self.select_file)
        layout.addWidget(self.btn_select)

        self.lang_box = QComboBox()
        self.lang_box.addItem("EN → RU")
        self.lang_box.addItem("RU → EN (future)")
        layout.addWidget(self.lang_box)

        self.btn_translate = QPushButton("Перевести")
        self.btn_translate.setEnabled(False)
        self.btn_translate.clicked.connect(self.start_translation)
        layout.addWidget(self.btn_translate)

        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        layout.addWidget(self.progress)

        self.output = QTextEdit()
        self.output.setReadOnly(True)
        layout.addWidget(self.output)

        self.setLayout(layout)

    # Style
    def apply_style(self):
        self.setStyleSheet(LIGHT_STYLE if self.theme_name == "light" else DARK_STYLE)

    def toggle_theme(self):
        if self.theme_name == "dark":
            self.theme_name = "light"
            self.btn_theme.setText("Тёмная тема")
        else:
            self.theme_name = "dark"
            self.btn_theme.setText("Светлая тема")
        self.apply_style()

    def _refresh_translate_button_style(self):
        self.btn_translate.style().unpolish(self.btn_translate)
        self.btn_translate.style().polish(self.btn_translate)

    # File select
    def select_file(self):

        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Выберите DOCX файл",
            "",
            "Documents (*.docx)"
        )

        if file_path:
            self.file_path = file_path
            self.file_label.setText(f"📄 {Path(file_path).name}")

    # Start / Cancel
    def start_translation(self):

        if self.controller and self.controller.state == AppState.TRANSLATING:
            self.controller.cancel_translation()
            return

        if self.controller and self.controller.state == AppState.CANCELLING:
            self.set_status("Отмена уже выполняется. Дождитесь остановки текущего батча.")
            return

        if not self.file_path:
            self.set_status("Файл не выбран")
            return

        self.controller.start_translation(self.file_path)

    def ask_save_path(self, suggested_path: str) -> str:
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Куда сохранить переведённый DOCX",
            suggested_path,
            "Documents (*.docx)"
        )
        return file_path or ""

    def clear_status(self):
        self.output.clear()

    def set_status(self, text: str):
        self.output.append(text)

    def set_progress(self, value: int):
        if self.progress.minimum() != 0 or self.progress.maximum() != 100:
            self.progress.setRange(0, 100)
        self.progress.setValue(value)

    def set_busy_progress(self, enabled: bool):
        if enabled:
            self.progress.setRange(0, 0)
        else:
            self.progress.setRange(0, 100)
            self.progress.setValue(0)

    def update_state(self, state: AppState):

        if state == AppState.LOADING:
            self.btn_translate.setObjectName("")
            self.btn_translate.setText("Модель загружается...")
            self.btn_translate.setEnabled(False)
            self.btn_select.setEnabled(True)

        elif state == AppState.TRANSLATING:
            self.btn_translate.setObjectName("cancelButton")
            self.btn_translate.setText("Отмена")
            self.btn_translate.setEnabled(True)
            self.btn_select.setEnabled(False)

        elif state == AppState.CANCELLING:
            self.btn_translate.setObjectName("cancelButton")
            self.btn_translate.setText("Отмена...")
            self.btn_translate.setEnabled(False)
            self.btn_select.setEnabled(False)

        elif state in [AppState.DONE, AppState.ERROR, AppState.IDLE]:
            is_ready = bool(self.controller and self.controller.model_ready)
            self.btn_translate.setObjectName("")
            self.btn_translate.setText("Перевести")
            self.btn_translate.setEnabled(is_ready)
            self.btn_select.setEnabled(True)

        self._refresh_translate_button_style()

if __name__ == "__main__":

    app = QApplication(sys.argv)

    window = TranslatorApp()
    window.show()

    sys.exit(app.exec_())
