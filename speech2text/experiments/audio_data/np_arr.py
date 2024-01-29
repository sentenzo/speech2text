from dataclasses import dataclass
from functools import lru_cache
from io import BytesIO
from os import PathLike
from typing import Any

import noisereduce  # can take quite some time
import numpy as np
import numpy.typing as np_typing
import torch  # can take quite some time
import whisper  # can take quite some time

from .audio_data import IAudioData, PcmParams
from .pd import PdData
from .wav import WavData

NP_DTYPE = "f2" if torch.cuda.is_available() else "f4"
DEFAULT_WHISPER_MODEL_NAME = (
    "small.en" if torch.cuda.is_available() else "tiny.en"
)


@lru_cache(3)
def _pick_whisper_model(model_name: str = DEFAULT_WHISPER_MODEL_NAME):
    return whisper.load_model(model_name, in_memory=True)


_pick_whisper_model()  # cold start overcoming


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

    def reduce_noise(self):
        new_data = noisereduce.reduce_noise(
            self._data,
            self.pcm_params.sample_rate,
            stationary=False,  # =False,
            y_noise=None,  # =None,
            prop_decrease=0.9,  # =1.0,
            time_constant_s=2.0,  # =2.0,
            freq_mask_smooth_hz=500,  # =500,
            time_mask_smooth_ms=50,  # =50,
            thresh_n_mult_nonstationary=2,  # =2,
            sigmoid_slope_nonstationary=10,  # =10,
            n_std_thresh_stationary=1.5,  # =1.5,
            tmp_folder=None,  # =None,
            chunk_size=600000,  # =600000,
            padding=30000,  # =30000,
            n_fft=1024,  # =1024,
            win_length=None,  # =None,
            hop_length=None,  # =None,
            clip_noise_stationary=True,  # =True,
            use_tqdm=False,  # =False,
            # n_jobs=-1,  # =1,
            use_torch=True,  # =False,
            device="cuda",  # ="cuda",
        )
        return NpData(self.pcm_params, new_data)

    def transcribe(
        self,
        model_name: str = DEFAULT_WHISPER_MODEL_NAME,
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
        return _pick_whisper_model(model_name).transcribe(
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
