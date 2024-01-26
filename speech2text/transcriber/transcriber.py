import whisper

from speech2text.utils.sample_types import SampleDType as Sdt
from speech2text.utils.samples import Samples


class Transcriber:
    def __init__(self):
        ...

    def transcribe(self, samples: Samples) -> str:
        raise NotImplementedError


class DummyTranscriber(Transcriber):
    def transcribe(self, samples: Samples) -> str:
        import time

        time.sleep(0.1)
        return "(dummy transcriber output) "


class WhisperTranscriber(Transcriber):
    def __init__(self):
        super().__init__()
        self.model = whisper.load_model("tiny.en", in_memory=True)

    def transcribe(self, samples: Samples) -> str:
        tr = whisper.transcribe(
            self.model,
            samples.as_type(Sdt.NP_F32),
            fp16=False,
            condition_on_previous_text=False,
        )
        return tr["text"]
