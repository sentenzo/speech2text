import whisper

from speech2text.transcriber.state import TranscriptionState as State

from . import Transformation


class Transcribe(Transformation):
    def __init__(self):
        ...

    def change_state(self, state: State):
        for block in state.blocks:
            np_array = block.data.as_np_float32()
            ...


# class WhisperTranscriber(Transcriber):
#     def __init__(self):
#         super().__init__()
#         self.model = whisper.load_model("tiny.en", in_memory=True)

#     def transcribe(self, samples: Samples) -> str:
#         tr = whisper.transcribe(
#             self.model,
#             samples.as_type(Sdt.NP_F32),
#             fp16=False,
#             condition_on_previous_text=False,
#         )
#         return tr["text"]
