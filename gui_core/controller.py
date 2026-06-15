from pathlib import Path

from gui_core.model_loader import ModelLoadWorker
from gui_core.worker import TranslateWorker
from gui_core.state import AppState


class AppController:

    def __init__(self, ui):
        self.ui = ui
        self.worker = None
        self.model_worker = None
        self.state = AppState.IDLE
        self.model_ready = False

    # Загрузка модели
    def start_model_preload(self):
        self.model_ready = False
        self.set_state(AppState.LOADING)
        self.ui.set_status("Загружаю модель при старте приложения...")
        self.ui.set_busy_progress(True)

        self.model_worker = ModelLoadWorker()
        self.model_worker.status.connect(self.on_model_status)
        self.model_worker.loaded.connect(self.on_model_loaded)
        self.model_worker.error.connect(self.on_model_error)
        self.model_worker.start()

    def on_model_status(self, text: str):
        self.ui.set_status(text)

    def on_model_loaded(self, text: str):
        self.model_ready = True
        self.ui.set_busy_progress(False)
        self.ui.set_progress(0)
        self.ui.set_status(text)
        self.ui.set_status("Теперь можно выбирать DOCX и запускать перевод.")
        self.set_state(AppState.IDLE)

    def on_model_error(self, msg: str):
        self.model_ready = False
        self.ui.set_busy_progress(False)
        self.set_state(AppState.ERROR)
        self.ui.set_status(f"Ошибка загрузки модели: {msg}")

    # Старт / Отмена перевода
    def start_translation(self, file_path: str):

        if not file_path:
            self.ui.set_status("Файл не выбран")
            return

        if not self.model_ready:
            self.ui.set_status("Модель ещё загружается. Дождитесь сообщения 'Модель готова'.")
            return

        if self.worker is not None and self.worker.isRunning():
            self.ui.set_status("Перевод уже выполняется. Для остановки нажмите 'Отмена'.")
            return

        self.set_state(AppState.TRANSLATING)

        if hasattr(self.ui, "clear_status"):
            self.ui.clear_status()
        self.ui.set_status("Перевод начался...")
        self.ui.set_progress(0)

        self.worker = TranslateWorker(file_path)
        self.worker.status.connect(self.on_worker_status)
        self.worker.progress.connect(self.on_progress)
        self.worker.document_ready.connect(self.on_document_ready)
        self.worker.cancelled.connect(self.on_cancelled)
        self.worker.error.connect(self.on_error)
        self.worker.start()

    def cancel_translation(self):
        if self.worker is None or not self.worker.isRunning():
            self.set_state(AppState.IDLE)
            return

        self.ui.set_status("Отменяю перевод... текущий батч завершится, затем процесс остановится.")
        self.set_state(AppState.CANCELLING)
        self.worker.request_cancel()

    def on_worker_status(self, text: str):
        self.ui.set_status(text)

    def on_progress(self, value: int):
        self.ui.set_progress(value)

    # Сохраненние
    def on_document_ready(self, doc, suggested_path: str):
        self.set_state(AppState.DONE)
        save_path = self.ui.ask_save_path(suggested_path)
        if not save_path:
            self.set_state(AppState.IDLE)
            self.ui.set_progress(0)
            self.ui.set_status("Сохранение отменено. Перевод не был записан в файл.")
            return

        path = Path(save_path)
        if path.suffix.lower() != ".docx":
            path = path.with_suffix(".docx")

        try:
            doc.save(str(path))
        except Exception as exc:
            self.on_error(f"Не удалось сохранить DOCX: {exc}")
            return

        self.set_state(AppState.DONE)
        self.ui.set_progress(100)
        self.ui.set_status(f"Готово. Файл сохранён: {path}")

    def on_cancelled(self, msg: str):
        self.set_state(AppState.IDLE)
        self.ui.set_status(msg or "Перевод отменён пользователем.")

    def on_error(self, msg: str):
        self.set_state(AppState.ERROR)
        self.ui.set_status(f"Ошибка: {msg}")


    def set_state(self, state: AppState):
        self.state = state
        self.ui.update_state(state)
