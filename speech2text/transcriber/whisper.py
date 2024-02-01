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
def _pick_whisper_model(model: ModelName | str = DEFAULT_WHISPER_MODEL_NAME):
    if isinstance(model, str):
        model = ModelName(model)
    return whisper.load_model(model.value, in_memory=True)


_pick_whisper_model()  # cold start overcoming


@dataclass(frozen=True)
class TranscriptionParameters:
    verbose: bool = False
    temperature: float | Tuple[float, ...] = (0, 0.2, 0.4, 0.6, 0.8, 1)
    compression_ratio_threshold: float = 2.4
    no_speech_threshold: float = 0.6
    condition_on_previous_text: bool = True
    initial_prompt: str | None = None
    word_timestamps: bool = False
    hallucination_silence_threshold: float | None = None

    def as_dict(self):
        return asdict(self)

    def replace(self, **changes):
        return replace(self, **changes)


DEFAULT_TRANSCRIPTION_PARAMETERS = TranscriptionParameters()


def transcribe(
    np_data: NpData,
    model_name: ModelName | str = DEFAULT_WHISPER_MODEL_NAME,
    params: TranscriptionParameters = DEFAULT_TRANSCRIPTION_PARAMETERS,
) -> dict[str, str | list]:
    if not isinstance(model_name, ModelName):
        model_name = ModelName(model_name)
    return _pick_whisper_model(model_name).transcribe(
        np_data._data,
        **params.as_dict(),
    )
