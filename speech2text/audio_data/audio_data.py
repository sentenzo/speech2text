import wave
from io import BytesIO
from os import PathLike
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
from IPython.display import Audio, display

from .pcm_params import PcmParams


class IAudioData:
    def create_io_stream(self) -> BytesIO:
        raise NotImplementedError

    @property
    def pcm_params(self) -> PcmParams:
        raise NotImplementedError

    @property
    def raw_data(self) -> bytes:
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

    def show_player(self):
        in_memory_file = self.create_io_stream()
        player = Audio(in_memory_file.read())
        display(player)

    def show_specgram(self, figsize=(14, 5), dpi=60):
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
