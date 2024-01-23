from multiprocessing import Queue

import pyaudio as pa

import speech2text.config as cfg

from .listener import Listener
from .pcm_params import PcmParams

SAMPLE_FORMATS = {2: pa.paInt16, 4: pa.paInt32}
BUFFER_SIZE_MULTI = 10


class PyAudioWrapper(pa.PyAudio):
    def __init__(self) -> None:
        super().__init__()
        self.inside_cm = False

    def __enter__(self):
        assert not self.inside_cm
        self.inside_cm = True
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.inside_cm = False
        self.terminate()


def _mic_recorder_proc(
    queue: Queue,
    pcm_params: PcmParams,
    chunk_size_sec: float,
    microphone_id: str,
) -> None:
    chunk_size_frames = pcm_params.seconds_to_sample_count(chunk_size_sec)
    buffer_size_frames = chunk_size_frames * BUFFER_SIZE_MULTI

    with PyAudioWrapper() as audio:
        stream = audio.open(
            input_device_index=microphone_id,
            format=SAMPLE_FORMATS[pcm_params.sample_width_bytes],
            channels=pcm_params.channels_count,
            rate=pcm_params.sample_rate,
            input=True,
            frames_per_buffer=buffer_size_frames,
        )
        try:
            while True:
                chunk = stream.read(num_frames=chunk_size_frames)
                queue.put(chunk)
        except KeyboardInterrupt:
            pass
        finally:
            stream.close()


class MicrophonListener(Listener):
    def __init__(self, *args, microphone_id: int = None):
        super().__init__(*args)
        with PyAudioWrapper() as audio:
            if microphone_id is None:
                microphone_id = audio.get_default_input_device_info()["index"]

        self.microphone_id = microphone_id
        self._recorder_proc_func = _mic_recorder_proc

    def _get_recorder_proc_args(self):
        pcm_params = self.pcm_params
        chunk_size_sec = cfg.CHUNK_SIZE_SEC
        microphone_id = self.microphone_id
        return (pcm_params, chunk_size_sec, microphone_id)
