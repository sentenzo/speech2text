from pathlib import Path
from typing import Tuple

import annotated_types
import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    PositiveFloat,
    PositiveInt,
    ValidationError,
    root_validator,
    validator,
)
from pydantic_settings import BaseSettings
from typing_extensions import Annotated

PositiveNormFloat = Annotated[
    float, annotated_types.Ge(0.0), annotated_types.Le(1.0)
]

Ge1Float = Annotated[float, annotated_types.Ge(1.0)]


class ListenerSettings(BaseModel):
    chunk_size_sec: PositiveFloat
    queue_check_delay_sec: PositiveFloat

    @root_validator(pre=True)
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
    low_pass_filter: PositiveInt | None = None
    high_pass_filter: PositiveInt | None = None
    volume_up: PositiveFloat | None = None
    speed_up: Ge1Float | None = None
    normalize: bool = False

    @root_validator(pre=True)
    def validate_date(cls, values):
        low_pass_filter = values["low_pass_filter"]
        high_pass_filter = values["high_pass_filter"]
        if low_pass_filter is not None and high_pass_filter is not None:
            if low_pass_filter >= high_pass_filter:
                raise ValidationError(
                    "Required: low_pass_filter < high_pass_filter\n"
                    f"Got: {low_pass_filter} >= {high_pass_filter}"
                )
        return values


class PyDubSplitOnSilenceSettings(BaseModel):
    min_silence_len: PositiveInt
    silence_thresh: int
    keep_silence: PositiveInt
    seek_step: PositiveInt


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
    temperature: PositiveNormFloat | Tuple[PositiveNormFloat, ...] = (
        0,
        0.2,
        0.4,
        0.6,
        0.8,
        1,
    )
    compression_ratio_threshold: PositiveFloat = 2.4
    no_speech_threshold: PositiveFloat = 0.6
    word_timestamps: bool = False
    hallucination_silence_threshold: PositiveFloat | None = None

    model_config = ConfigDict(protected_namespaces=())

    @validator("temperature", pre=True)
    def validate_temperature(cls, temperature):
        if isinstance(temperature, list):
            return tuple(temperature)
        return temperature


class IncrementStageSettings(BaseModel):
    pass


class AdjustStageSettings(BaseModel):
    pydub: PyDubSettings


class SplitStageSettings(BaseModel):
    class AgressiveThreshold(BaseModel):
        latency_ratio: PositiveFloat
        duration_sec: PositiveFloat

    class SplitOnSilence(BaseModel):
        default: PyDubSplitOnSilenceSettings
        agressive_default: PyDubSplitOnSilenceSettings
        very_agressive_default: PyDubSplitOnSilenceSettings

    agressive_threshold: AgressiveThreshold
    pudub_split_on_silence: SplitOnSilence


class RefineStageSettings(BaseModel):
    class SubSection(BaseModel):
        pydub: PyDubSettings | None
        noisereduce: NoisereduceSettings | None

    ongoing: SubSection
    final: SubSection


class TranscribeStageSettings(BaseModel):
    class SubSection(BaseModel):
        whisper: WhisperSettings

    ongoing: SubSection
    final: SubSection


class AllStageSettings(BaseModel):
    increment: IncrementStageSettings | None
    adjust: AdjustStageSettings
    split: SplitStageSettings
    refine: RefineStageSettings
    transcribe: TranscribeStageSettings


class Settings(BaseSettings):
    class StagesSubSection(BaseModel):
        stages: AllStageSettings

    listener: ListenerSettings | None
    transcriber: StagesSubSection | None


APP_DIR = Path(__file__).parent.parent


def load_settings_from_yaml(yaml_file_path: str):
    with open(yaml_file_path, "r") as file:
        yaml_data = yaml.safe_load(file)

    return Settings(**yaml_data)


app_settings = load_settings_from_yaml(APP_DIR / "config.yaml")
