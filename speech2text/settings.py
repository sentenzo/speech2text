from typing import List, Tuple

import annotated_types
from pydantic import (
    BaseModel,
    Field,
    PositiveFloat,
    PositiveInt,
    ValidationError,
    root_validator,
)
from pydantic_settings import BaseSettings
from typing_extensions import Annotated

PositiveNormFloat = Annotated[
    float, annotated_types.Ge(0.0), annotated_types.Le(1.0)
]


class ListenerSettings(BaseModel):
    chunk_size_sec: PositiveFloat
    queue_check_delay_sec: PositiveFloat

    @root_validator
    def validate_date(cls, values):
        queue_check_delay_sec = values["queue_check_delay_sec"]
        chunk_size_sec = values["chunk_size_sec"]
        if queue_check_delay_sec >= chunk_size_sec:
            raise ValidationError(
                "Required: queue_check_delay_sec >= chunk_size_sec\n"
                f"Got: {queue_check_delay_sec} < {chunk_size_sec}"
            )
        return values


class PyDubSettings(BaseModel):
    low_pass_filter: PositiveInt | None
    high_pass_filter: PositiveInt | None
    volume_up: PositiveFloat = Field(1.0)
    speed_up: float = Field(1.0, ge=1.0)
    normalize: bool = False

    @root_validator
    def validate_date(cls, values):
        low_pass_filter = values["low_pass_filter"]
        high_pass_filter = values["high_pass_filter"]
        if low_pass_filter >= high_pass_filter:
            raise ValidationError(
                "Required: low_pass_filter < high_pass_filter\n"
                f"Got: {low_pass_filter} >= {high_pass_filter}"
            )
        return values


class NoisereduceSettings(BaseModel):
    stationary: bool
    prop_decrease: PositiveNormFloat
    freq_mask_smooth_hz: PositiveInt
    time_mask_smooth_ms: PositiveInt
    chunk_size: PositiveInt
    padding: PositiveInt
    n_fft: PositiveInt
    clip_noise_stationary: bool
    use_tqdm: bool


class WhisperSettings(BaseModel):
    model_name: str
    temperature: (
        PositiveNormFloat
        | Tuple[PositiveNormFloat, ...]
        | List[PositiveNormFloat]
    ) = (0, 0.2, 0.4, 0.6, 0.8, 1)
    compression_ratio_threshold: PositiveFloat = 2.4
    no_speech_threshold: PositiveFloat = 0.6
    word_timestamps: bool = False
    hallucination_silence_threshold: PositiveFloat | None = None


class Settings(BaseSettings):
    ...
