import logging
import wave

import noisereduce as nr
import numpy as np
from pydub import AudioSegment, effects, silence

BITS_IN_BYTE = 8
SAMPLE_RATE = 16_000
SAMPLE_WIDTH = 2

logger = logging.getLogger(__name__)


def convert(in_path, out_path):
    logger.debug("start converting")
    wav_file: AudioSegment = AudioSegment.from_wav(in_path)
    wav_file = wav_file.set_channels(1)  # to mono
    wav_file = wav_file.set_frame_rate(SAMPLE_RATE)

    samples = wav_file.get_array_of_samples()
    samples = np.array(samples).astype(np.float32)
    samples /= 1 << 15
    samples = nr.reduce_noise(samples, SAMPLE_RATE)
    samples = np.array(samples * (1 << 15), dtype=np.int16)
    wav_file = wav_file._spawn(samples)

    # wav_file
    wav_file.export(
        out_path,
        format="wav",
    )
    logger.debug("stop converting")


def split(in_path, out_path):
    audio: AudioSegment = AudioSegment.from_wav(in_path)
    audio = effects.normalize(audio)
    # audio += 0

    segments = silence.split_on_silence(
        audio,
        min_silence_len=500,
        silence_thresh=-30,
        keep_silence=300,
        seek_step=10,  # default was 1 ms
    )
    for i, segment in enumerate(segments):
        segment.export(
            out_path + f"{i}.wav",
            format="wav",
        )
