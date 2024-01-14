import time

import pyaudio as pa


class Listener:
    async def chunks(chunk_duration_sec: float):
        while True:
            time.sleep(chunk_duration_sec)


class MicrophoneListener(Listener):
    def __init__(self, device_id=None) -> None:
        pa.PyAudio().open()


class FileListener(Listener):
    def __init__(self) -> None:
        super().__init__()
