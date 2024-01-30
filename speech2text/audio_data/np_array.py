from dataclasses import dataclass
from io import BytesIO
from os import PathLike

import numpy as np
import numpy.typing as np_typing
from torch.cuda import is_available as is_cuda_available

from .audio_data import IAudioData, PcmParams
from .pydub_audioseg import PdData
from .wave_data import WavData

NP_DTYPE = "f2" if is_cuda_available() else "f4"


@dataclass
class NpData(IAudioData):
    """An `IAudioData` implementation for `np.float`-arrays (2 or 4 bytes)."""

    _pcm_params: PcmParams
    _data: np_typing.NDArray[np.float32]

    @staticmethod
    def load_from_wav_file(
        wav_file: str | bytes | PathLike | WavData,
    ) -> "NpData":
        pcm_params = data = None
        if isinstance(wav_file, WavData):
            pcm_params = wav_file.pcm_params
            data = wav_file.raw_data
        else:
            pcm_params, data = NpData._load_from_wav_file(wav_file)

        dtype = f"<i{pcm_params.sample_width_bytes}"
        np_array: np_typing.NDArray = np.frombuffer(data, dtype)
        max_val = 2 ** (pcm_params.sample_width_bytes * 8 - 1)
        np_array = np_array.astype(NP_DTYPE) / max_val

        return NpData(pcm_params, np_array)

    @staticmethod
    def load_from_pd_data(audio: PdData) -> "NpData":
        return NpData.load_from_wav_file(audio.create_io_stream())

    def create_io_stream(self) -> BytesIO:
        wav_data = WavData(self.pcm_params, self.raw_data)
        return wav_data.create_io_stream()

    @property
    def pcm_params(self) -> PcmParams:
        return self._pcm_params

    @property
    def raw_data(self) -> bytes:
        np_array: np_typing.NDArray = np.copy(self._data)
        sample_width_bytes = self.pcm_params.sample_width_bytes
        max_val = 2 ** (sample_width_bytes * 8 - 1)
        np_array *= max_val
        np.clip(np_array, -max_val, max_val - 1)
        data = np_array.astype(f"<i{sample_width_bytes}").tobytes()
        return data
