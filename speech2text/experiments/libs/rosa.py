import logging
import wave

import librosa
import noisereduce as nr
import numpy as np

BITS_IN_BYTE = 8
SAMPLE_RATE = 16_000
SAMPLE_WIDTH = 2

logger = logging.getLogger(__name__)


def convert(in_path, out_path):
    logger.log(15, "start converting")
    samples, _ = librosa.load(
        in_path,
        sr=SAMPLE_RATE,
        mono=True,
        res_type="soxr_vhq",  # ‘soxr_vhq’, ‘soxr_hq’, ‘soxr_mq’ or ‘soxr_lq’
    )
    samples = nr.reduce_noise(samples, SAMPLE_RATE)
    out_file: wave.Wave_write
    with wave.open(out_path, "wb") as out_file:
        out_file.setnchannels(1)
        out_file.setframerate(SAMPLE_RATE)
        out_file.setsampwidth(SAMPLE_WIDTH)

        samples *= 1 << (BITS_IN_BYTE * SAMPLE_WIDTH - 1)
        samples = np.int16(samples)
        out_file.writeframes(samples.tobytes())

    logger.log(15, "stop converting")
