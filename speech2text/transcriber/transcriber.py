from whisper import transcribe

from speech2text.utils.samples import Samples


class Transcriber:
    def __init__(self):
        ...

    def transcribe(self, samples: Samples) -> str:
        raise NotImplementedError


class DummyTranscriber:
    def transcribe(self, samples: Samples) -> str:
        import time

        time.sleep(0.1)
        return "(dummy transcriber output) "
