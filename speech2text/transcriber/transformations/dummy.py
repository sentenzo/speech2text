import time

from speech2text.transcriber.state import (
    TranscriptionState as State,
    # TranscriptionStateTransformation as Transformation,
)
from ..transformations import Transformation


class Dummy(Transformation):
    def __init__(self, delay: float):
        self.delay = delay

    def change_state(self, state: State):
        time.sleep(self.delay)
