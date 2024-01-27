import logging

from speech2text.experiments.dub import split as pydub_split

logger = logging.getLogger(__name__)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s \t|\t%(message)s",
)

IN_FILE_PATH = "tests/audio_samples/en_hedgehog.wav"
OUT_FILE_PATH = "tests/audio_samples/out.wav"


def mvp():
    from .listener import MicrophonListener, WavFileListener
    from .transcriber import Workflow

    listener = WavFileListener(path_to_file=IN_FILE_PATH)
    # listener = MicrophonListener()
    wfq = Workflow()
    try:
        for chunk in listener.get_chunks_iterator():
            logger.debug(
                f"A chunk of size {len(chunk)} bytes is received. "
                f"The 123-th value is: {chunk[123]}"
            )
            wfq.push_chunk(chunk)
            # print("\r" + " " * 80 + "\r")
            for line in wfq.flush_finalized_text():
                print(line)
            print("\r", wfq.get_transcription(), " " * 10, end="")
    except KeyboardInterrupt:
        pass


def test_file_listener():
    import wave

    from .listener import WavFileListener

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
    import wave

    from .listener import MicrophonListener

    listener = MicrophonListener()
    with wave.open(OUT_FILE_PATH, "wb") as wav_file:
        wav_file.setparams(listener.pcm_params.wav_params)

        for chunk in listener.get_chunks_iterator():
            logger.debug(
                f"A chunk of size {len(chunk)} bytes is received. "
                f"The 123-th value is: {chunk[123]}"
            )
            wav_file.writeframes(chunk)


if __name__ == "__main__":
    mvp()
    # pydub_split(IN_FILE_PATH, OUT_FILE_PATH)
