from enum import Enum, auto
from speech2text.pcm_params import PcmParams, WHISPER_PRESET
import numpy as np
import numpy.typing as np_typing
from pydub import AudioSegment

POW_2_15 = 1 << 15


class SamplesFormat(Enum):
    BINARY = auto()
    PD_A_SEGMENT = auto()  # PyDub AudioSegment
    NP_FLOAT32 = auto()  # np_typing.NDArray[np.float32]


class Samples:
    def __init__(
        self,
        data: bytes | AudioSegment | np_typing.NDArray[np.float32],
        sample_format: SamplesFormat,
        pcm_params: PcmParams = WHISPER_PRESET,
    ) -> None:
        self.data = data
        self.sample_format = sample_format
        self.pcm_params = pcm_params

    def _as_binary(self) -> "Samples":
        if self.sample_format == SamplesFormat.BINARY:
            return self
        if self.sample_format == SamplesFormat.PD_A_SEGMENT:
            self.data = bytes(self.data.get_array_of_samples())
            self.sample_format = SamplesFormat.BINARY
            return self
        if self.sample_format == SamplesFormat.NP_FLOAT32:
            np_array = self.data
            np_array *= POW_2_15
            np.clip(np_array, -POW_2_15, POW_2_15 - 1)
            self.data = np_array.astype("<i2").tobytes()
            self.sample_format = SamplesFormat.BINARY
            return self

    def as_binary(self) -> bytes:
        return self._as_binary().data

    def _as_pd_a_segment(self) -> "Samples":
        if self.sample_format == SamplesFormat.BINARY:
            pcm_params = self.pcm_params
            pcm_kwargs = {
                "channels": pcm_params.channels_count,
                "sample_width": pcm_params.sample_width_bytes,
                "frame_rate": pcm_params.sample_rate,
            }
            audio: AudioSegment = AudioSegment(self.data, **pcm_kwargs)
            self.data = audio
            self.sample_format = SamplesFormat.PD_A_SEGMENT
            return self
        if self.sample_format == SamplesFormat.PD_A_SEGMENT:
            return self
        if self.sample_format == SamplesFormat.NP_FLOAT32:
            # self.sample_format = SamplesFormat.PD_A_SEGMENT
            return self._as_binary()._as_pd_a_segment()

    def as_pd_a_segment(self) -> AudioSegment:
        return self._as_pd_a_segment().data

    def _as_np_float32(self) -> "Samples":
        if self.sample_format == SamplesFormat.BINARY:
            np_format_str = (
                f"<i{self.pcm_params.sample_width_bytes}"  # "<i2" or "<i4"
            )
            np_array: np_typing.NDArray = np.frombuffer(
                self.data, np_format_str
            )
            np_array = np_array.astype("f4") / POW_2_15
            self.data = np_array
            self.sample_format = SamplesFormat.NP_FLOAT32
            return self
        if self.sample_format == SamplesFormat.PD_A_SEGMENT:
            # self.sample_format = SamplesFormat.NP_FLOAT32
            return self._as_binary()._as_np_float32()
        if self.sample_format == SamplesFormat.NP_FLOAT32:
            return self

    def _as_np_float32(self) -> np_typing.NDArray[np.float32]:
        return self._as_np_float32().data
