from functools import lru_cache

import noisereduce  # can take quite some time
import whisper  # can take quite some time
from torch.cuda import is_available as is_cuda_available

from speech2text.audio_data import NpData

DEFAULT_WHISPER_MODEL_NAME = "small.en" if is_cuda_available() else "tiny.en"


@lru_cache(3)
def _pick_whisper_model(model_name: str = DEFAULT_WHISPER_MODEL_NAME):
    return whisper.load_model(model_name, in_memory=True)


_pick_whisper_model()  # cold start overcoming


def reduce_noise(np_data: NpData) -> NpData:
    new_data = noisereduce.reduce_noise(
        np_data._data,
        np_data.pcm_params.frame_rate,
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
    return NpData(np_data.pcm_params, new_data)


def transcribe(
    np_data: NpData,
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
        np_data._data,
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
