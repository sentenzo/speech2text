import logging

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

    # listener = WavFileListener(path_to_file=IN_FILE_PATH)
    listener = MicrophonListener()
    wfq = Workflow()
    for chunk in listener.get_chunks_iterator():
        try:
            logger.debug(
                f"A chunk of size {len(chunk)} bytes is received. "
                f"The 123-th value is: {chunk[123]}"
            )
            wfq.push_chunk(chunk)
            # print("\r" + " " * 80 + "\r")
            for line in wfq.flush_text():
                print(line)
            print("\r", str(wfq), " " * 10, end="")
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
    # test_file_listener()
    from .transcriber import Workflow
    from .transcriber.transformations import Dummy

    t1 = Dummy(0.05)
    t2 = Dummy(0.05)
    t3 = Dummy(0.05)

    wf = Workflow(t1 >> t2 >> t3)
