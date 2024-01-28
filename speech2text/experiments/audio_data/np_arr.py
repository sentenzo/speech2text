from dataclasses import dataclass
from io import BytesIO
from os import PathLike

import numpy as np
import numpy.typing as np_typing

from . import IAudioData, PcmParams, PdData, WavData

NP_DTYPE = "f4"


@dataclass
class NpData(IAudioData):
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
        np_array = np_array.astype("f4") / max_val

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

    @classmethod
    def _load_noisereduce_module(cls):
        """Can sometimes take up to 2 min"""
        import noisereduce

        cls.noisereduce = noisereduce

    def reduce_noise(self):
        if not hasattr(self, "noisereduce"):
            self._load_noisereduce_module()

        new_data = self.noisereduce.reduce_noise(
            self._data, self.pcm_params.sample_rate
        )
        return NpData(self.pcm_params, new_data)

    @classmethod
    def _load_whisper_model(cls, model_name: str = "small.en"):
        """small.en can sometimes take up to 20 sec"""

        import whisper

        cls.whisper_model = whisper.load_model(model_name, in_memory=True)

    def transcribe(
        self,
        verbose=False,
        temperature=(0, 0.2, 0.4, 0.6, 0.8, 1),
        compression_ratio_threshold=2.4,
        no_speech_threshold=0.6,
        condition_on_previous_text=True,
        initial_prompt=None,
        word_timestamps=False,
        clip_timestamps="0",  # ??
        hallucination_silence_threshold=None,
    ) -> dict[str, str | list]:
        if not hasattr(self, "whisper_model"):
            self._load_whisper_model()
        return self.whisper_model.transcribe(
            self._data,
            verbose=verbose,
            temperature=temperature,
            compression_ratio_threshold=compression_ratio_threshold,
            no_speech_threshold=no_speech_threshold,
            condition_on_previous_text=condition_on_previous_text,
            initial_prompt=initial_prompt,
            word_timestamps=word_timestamps,
            clip_timestamps=clip_timestamps,
            hallucination_silence_threshold=hallucination_silence_threshold,
        )
