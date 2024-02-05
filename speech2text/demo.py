import logging

logger = logging.getLogger(__name__)


def demo_file_listener(in_file_path, out_file_path):
    """Reads from `in_file_path` (wav-file).

    Writes to `out_file_path` (wav-file).
    """
    import wave

    from .listener import WavFileListener

    listener = WavFileListener(path_to_file=in_file_path)
    with wave.open(out_file_path, "wb") as wav_file:
        wav_file.setparams(listener.pcm_params.wav_params)

        for chunk in listener.get_chunks_iterator():
            logger.debug(
                f"A chunk of size {len(chunk)} bytes is received. "
                f"The 123-th value is: {chunk[123]}"
            )
            wav_file.writeframes(chunk)


def demo_mic_listener(out_file_path):
    """Reads from the default input audio device (microphone).

    Writes to `out_file_path` (wav-file).
    """
    import wave

    from .listener import MicrophonListener

    listener = MicrophonListener()
    with wave.open(out_file_path, "wb") as wav_file:
        wav_file.setparams(listener.pcm_params.wav_params)

        for chunk in listener.get_chunks_iterator():
            logger.debug(
                f"A chunk of size {len(chunk)} bytes is received. "
                f"The 123-th value is: {chunk[123]}"
            )
            wav_file.writeframes(chunk)


def demo_console_realtime(in_file_path=None):
    from .audio_data import WHISPER_PCM_PARAMS
    from .listener import MicrophonListener, WavFileListener
    from .transcriber import Workflow

    listener = MicrophonListener(WHISPER_PCM_PARAMS)
    if in_file_path:
        listener = WavFileListener(
            WHISPER_PCM_PARAMS, path_to_file=in_file_path
        )
    wfq = Workflow(input_pcm_params=WHISPER_PCM_PARAMS)
    try:
        print("\033[H\033[J", end="")  # clear the entire console
        for chunk in listener.get_chunks_iterator(0.8):
            wfq.process_chunk(chunk)
            print("\033[A\033[0K", end="")  # clear the last line
            for line in wfq.get_finalized_text(flush_blocks=True):
                print("::", line)
            print(">>", wfq.get_ongoing_text())
    except KeyboardInterrupt:
        pass
