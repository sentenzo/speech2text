import whisper

from speech2text.transcriber.state import TranscriptionState as State

from . import Transformation


class Transcribe(Transformation):
    def __init__(self):
        self.model = whisper.load_model("tiny.en", in_memory=True)

    def change_state(self, state: State):
        initial_prompt = None
        if state.text_finalized:
            initial_prompt = state.text_finalized[-1]
        for block in state.blocks:
            np_array = block.samples.as_np_float32()
            transcription = whisper.transcribe(
                self.model,
                np_array,
                fp16=False,
                condition_on_previous_text=False,
                initial_prompt=initial_prompt,
            )
            block.text = transcription["text"]
            initial_prompt = block.text


class Finalize(Transformation):
    def change_state(self, state: State):
        if len(state.blocks) > 1:
            last_block = state.blocks.pop()
            for block in state.blocks:
                state.text_finalized.append(block.text)
            state.blocks = [last_block]
