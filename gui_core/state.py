from enum import Enum


class AppState(Enum):
    IDLE = "idle"
    LOADING = "loading"
    TRANSLATING = "translating"
    CANCELLING = "cancelling"
    DONE = "done"
    ERROR = "error"
