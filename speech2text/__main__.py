import logging
import wave

from .listener import MicrophonListener, WavFileListener

logger = logging.getLogger(__name__)

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(name)s \t|\t%(message)s",
)


if __name__ == "__main__":
    in_file_path = "tests/audio_samples/en_hedgehog.wav"
    out_file_path = "tests/audio_samples/out.wav"

    # listener = WavFileListener(path_to_file=in_file_path)
    listener = MicrophonListener()
    with wave.open(out_file_path, "wb") as wav_file:
        wav_file.setparams(listener.pcm_params.wav_params)

        for chunk in listener.get_chunks_iterator():
            logger.debug(
                f"A chunk of size {len(chunk)} bytes is received. "
                f"The 123-th value is: {chunk[123]}"
            )
            wav_file.writeframes(chunk)
