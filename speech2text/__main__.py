import logging

from speech2text.demo import (
    demo_console_realtime,
    demo_file_listener,
    demo_mic_listener,
)

logger = logging.getLogger(__name__)

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s - %(levelname)s - %(name)s \t|\t%(message)s",
)

IN_FILE_PATH = "tests/audio_samples/en_chunk.wav"
OUT_FILE_PATH = "tests/audio_samples/out.wav"


if __name__ == "__main__":
    demo_console_realtime(IN_FILE_PATH)
