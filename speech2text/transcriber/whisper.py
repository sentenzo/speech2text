from dataclasses import asdict, dataclass, replace
from enum import Enum
from functools import lru_cache
from typing import Tuple

import whisper  # can take quite some time
from torch.cuda import is_available as is_cuda_available

from speech2text.audio_data import NpData


class ModelName(Enum):
    TINY = "tiny"
    TINY_EN = "tiny.en"
    SMALL = "small"
    SMALL_EN = "small.en"


DEFAULT_WHISPER_MODEL_NAME = (
    ModelName.SMALL_EN if is_cuda_available() else ModelName.TINY_EN
)


@lru_cache(3)
def _pick_whisper_model(
    model_name: ModelName | str = DEFAULT_WHISPER_MODEL_NAME,
):
    if isinstance(model_name, str):
        model_name = ModelName(model_name)
    return whisper.load_model(model_name.value, in_memory=True)


_pick_whisper_model()  # cold start overcoming


@dataclass(frozen=True)
class TranscriptionParameters:
    verbose: bool
    temperature: float | Tuple[float, ...]
    compression_ratio_threshold: float
    no_speech_threshold: float
    condition_on_previous_text: bool
    initial_prompt: str | None
    word_timestamps: bool
    hallucination_silence_threshold: float | None

    def as_dict(self):
        return asdict(self)

    def replace(self, **changes):
        return replace(self, **changes)


DEFAULT_TRANSCRIPTION_PARAMETERS = TranscriptionParameters(
    verbose=None,
    temperature=(0, 0.2, 0.4, 0.6, 0.8, 1),
    compression_ratio_threshold=2.4,
    no_speech_threshold=0.6,
    condition_on_previous_text=True,
    initial_prompt=None,
    word_timestamps=False,
    hallucination_silence_threshold=None,
)


def transcribe(
    np_data: NpData,
    model_name: ModelName | str = DEFAULT_WHISPER_MODEL_NAME,
    # params: TranscriptionParameters = DEFAULT_TRANSCRIPTION_PARAMETERS,
    **kwargs,
) -> dict[str, str | list]:
    # if not isinstance(model_name, ModelName):
    #     model_name = ModelName(model_name)
    return _pick_whisper_model(model_name).transcribe(
        np_data._data,
        **kwargs,
        # **params.as_dict(),
    )
