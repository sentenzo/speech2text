import logging
import wave

from .listener import MicrophonListener, WavFileListener
from .transcriber import WorkflowQueue

logger = logging.getLogger(__name__)

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(name)s \t|\t%(message)s",
)

IN_FILE_PATH = "tests/audio_samples/en_123.wav"
OUT_FILE_PATH = "tests/audio_samples/out.wav"


if __name__ == "__main__":
    listener = WavFileListener(path_to_file=IN_FILE_PATH)
    wfq = WorkflowQueue()
    for chunk in listener.get_chunks_iterator():
        logger.debug(
            f"A chunk of size {len(chunk)} bytes is received. "
            f"The 123-th value is: {chunk[123]}"
        )
        wfq.push_chunk(chunk)
        print("\r" + " " * 80 + "\r")
        for line in wfq.flush_text():
            print(line)
        print(str(wfq))


def test_file_listener():
    listener = WavFileListener(path_to_file=IN_FILE_PATH)
    with wave.open(OUT_FILE_PATH, "wb") as wav_file:
        wav_file.setparams(listener.pcm_params.wav_params)

        for chunk in listener.get_chunks_iterator():
            logger.debug(
                f"A chunk of size {len(chunk)} bytes is received. "
                f"The 123-th value is: {chunk[123]}"
            )
            wav_file.writeframes(chunk)


def test_mic_listener():
    listener = MicrophonListener()
    with wave.open(OUT_FILE_PATH, "wb") as wav_file:
        wav_file.setparams(listener.pcm_params.wav_params)

        for chunk in listener.get_chunks_iterator():
            logger.debug(
                f"A chunk of size {len(chunk)} bytes is received. "
                f"The 123-th value is: {chunk[123]}"
            )
            wav_file.writeframes(chunk)
