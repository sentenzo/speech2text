import logging

from speech2text.audio_data import pcm_params

logger = logging.getLogger(__name__)

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s - %(levelname)s - %(name)s \t|\t%(message)s",
)

IN_FILE_PATH = "tests/audio_samples/en_chunk.wav"
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
    from .audio_data import WHISPER_PCM_PARAMS
    from .listener import MicrophonListener, WavFileListener
    from .transcriber import Workflow

    listener = WavFileListener(WHISPER_PCM_PARAMS, path_to_file=IN_FILE_PATH)
    # listener = MicrophonListener(WHISPER_PCM_PARAMS)
    wfq = Workflow(input_pcm_params=WHISPER_PCM_PARAMS)
    try:
        print("\033[H\033[J", end="")
        for chunk in listener.get_chunks_iterator(0.8):
            wfq.process_chunk(chunk)
            print("\033[A\033[A")
            for line in wfq.get_finalized_text(flush_blocks=True):
                print("::", line, " " * 5)
            print(">>", wfq.get_ongoing_text())
    except KeyboardInterrupt:
        pass
