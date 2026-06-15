from PyQt5.QtCore import QThread, pyqtSignal


class ModelLoadWorker(QThread):
    """Loads the model once at application startup without blocking the GUI."""

    loaded = pyqtSignal(str)
    error = pyqtSignal(str)
    status = pyqtSignal(str)

    def run(self):
        try:
            from src.inference import preload_model, get_runtime_info

            preload_model(status_callback=self.status.emit)
            self.loaded.emit(get_runtime_info())
        except Exception as exc:
            self.error.emit(str(exc))
