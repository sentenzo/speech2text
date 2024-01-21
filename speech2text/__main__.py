import logging

from .experiments.dub import convert as dub_convert
from .experiments.rosa import convert as rosa_convert
from .experiments.wave_io import WaveStream

logging.basicConfig(
    level=15,
    format="%(asctime)s - %(levelname)s \t|\t%(message)s",
)

if __name__ == "__main__":
    in_file_path = "tests/audio_samples/en_crows.wav"
    out_file_path = "tests/audio_samples/out.wav"
    # wave_stream = WaveStream(in_file_path)
    # wave_stream.to_mono().write_all_to_file(out_file_path)

    # rosa_convert(in_file_path, out_file_path + ".rosa.wav")
    dub_convert(in_file_path, out_file_path + ".dub.wav")

    # rosa_convert(in_file_path, out_file_path + ".rosa.wav")
    # rosa_convert(in_file_path, out_file_path + ".rosa.wav")
