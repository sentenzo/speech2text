import time

from .state import TranscriptionState as State
from .state import TranscriptionStateTransformation as Transformation


class Dummy(Transformation):
    def __init__(self, delay: float):
        self.delay = delay

    def change_state(self, state: State):
        time.sleep(self.delay)
