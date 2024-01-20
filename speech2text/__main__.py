import logging

from .experiments.wave_io import WaveStream

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s \t|\t%(message)s",
)

if __name__ == "__main__":
    in_file_path = "tests/audio_samples/en_crows.wav"
    out_file_path = "tests/audio_samples/out.wav"
    wave_stream = WaveStream(in_file_path)
    wave_stream.set_nchannels(1).write_all_to_file(out_file_path)
