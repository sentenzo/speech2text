import wave
from io import BytesIO
from os import PathLike
from typing import Tuple

import numpy as np

from .pcm_params import PcmParams


class IAudioData:
    """An interface to work with big PCM-encoded chunks of audio (from 0.1 to
    200 sec)."""

    def create_io_stream(self) -> BytesIO:
        """Creates an in-memory WAVE-file representation of the audio."""
        raise NotImplementedError

    @property
    def pcm_params(self) -> PcmParams:
        raise NotImplementedError

    @property
    def raw_data(self) -> bytes:
        """Returns a PCM encoded WAVE representation, but only the frames."""
        raise NotImplementedError

    def _load_from_wav_file(
        wav_file: str | bytes | PathLike,
    ) -> Tuple[PcmParams, bytearray]:
        if isinstance(wav_file, PathLike):
            wav_file = str(wav_file)
        with wave.open(wav_file, "rb") as file:
            pcm_params = PcmParams.from_wave_file(file)
            data = bytearray(file.readframes(file.getnframes()))
            return (pcm_params, data)

    def ipy_show_player(self):
        """In Jupyter Notebook: shows an interactive audio player element."""
        from IPython.display import Audio, display

        in_memory_file = self.create_io_stream()
        player = Audio(in_memory_file.read())
        display(player)

    def ipy_show_specgram(self, figsize=(14, 5), dpi=60):
        """In Jupyter Notebook: draws a spectrogram of the audio."""
        import matplotlib.pyplot as plt

        dtype = f"<i{self.pcm_params.sample_width_bytes}"
        data = np.frombuffer(self.raw_data, dtype=dtype)
        if self.pcm_params.channels_count > 1:
            data = np.split(data, self.pcm_params.channels_count)
            data = np.sum(data) / self.pcm_params.channels_count

        plt.figure(figsize=figsize, dpi=dpi)
        plt.specgram(
            data,
            Fs=self.pcm_params.frame_rate,
            NFFT=1024,
            cmap="magma",
        )
        # https://matplotlib.org/stable/users/explain/colors/colormaps.html#qualitative
        plt.xlabel("Time (s)")
        plt.ylabel("Frequency (Hz)")
        plt.colorbar(label="Intensity (dB)")
        plt.show()
