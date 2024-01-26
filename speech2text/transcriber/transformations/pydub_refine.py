from pydub import AudioSegment, effects

from speech2text.transcriber.state import (
    TranscriptionState as State,
)
from ..transformations import Transformation


class PdNormalize(Transformation):
    def change_state(self, state: State):
        for block in state.blocks:
            audio: AudioSegment = block.data.as_pd_a_segment()
            audio = effects.normalize(audio)
            samples = audio.get_array_of_samples()
            block.data = samples


class PdSplitSilence(Transformation):
    ...


class PdMaybeSpeedUp(Transformation):
    ...
