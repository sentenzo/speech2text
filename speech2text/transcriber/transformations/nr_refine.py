import noisereduce as nr

from speech2text.transcriber.state import TranscriptionState as State

from . import Transformation


class NoiseReduce(Transformation):
    def __init__(self) -> None:
        pass

    def change_state(self, state: State):
        for block in state.blocks:
            np_array = block.data.as_np_float32()
            np_array = nr.reduce_noise(np_array, state.pcm_params.sample_rate)
            block.data = np_array
