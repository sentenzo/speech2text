import logging
import wave

import pyaudio

logger = logging.getLogger(__name__)

RATE = 16_000
OUT_PATH = "tests/audio_samples/out.pyaudio.wav"
CHANNELS = 1
S_WIDTH = 2
BUFFER_SIZE = 1024


def rec(sec=3.0):
    audio = pyaudio.PyAudio()
    format = {2: pyaudio.paInt16, 4: pyaudio.paInt32}
    stream = audio.open(
        format=format[S_WIDTH],
        channels=CHANNELS,
        rate=RATE,
        input=True,
        frames_per_buffer=BUFFER_SIZE,
    )
    data = bytearray()
    times = int(float(sec) * RATE / BUFFER_SIZE)

    logger.log(15, "start recording")
    for _ in range(times):
        data.extend(stream.read(BUFFER_SIZE))
    logger.log(15, "stop recording")

    with wave.open(OUT_PATH, "wb") as out_file:
        out_file.setframerate(RATE)
        out_file.setnchannels(CHANNELS)
        out_file.setsampwidth(S_WIDTH)
        out_file.writeframes(data)
