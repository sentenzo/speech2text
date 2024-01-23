import logging
import wave

import speech2text.config as cfg

from .listener import MicrophonListener, WaveFileListener

logger = logging.getLogger(__name__)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s \t|\t%(message)s",
)


if __name__ == "__main__":
    in_file_path = "tests/audio_samples/en_hedgehog.wav"
    out_file_path = "tests/audio_samples/out.wav"

    listener = WaveFileListener(in_file_path)
    # listener = MicrophonListener()
    with wave.open(out_file_path, "wb") as wav_file:
        wav_file.setnchannels(cfg.CHANNELS)
        wav_file.setsampwidth(cfg.SAMPLE_WIDTH)
        wav_file.setframerate(cfg.SAMPLE_RATE)

        for chunk in listener.gen_chunks():
            wav_file.writeframes(chunk)
