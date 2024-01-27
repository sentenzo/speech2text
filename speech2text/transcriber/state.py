from dataclasses import dataclass, field

from speech2text.pcm_params import WHISPER_PRESET, PcmParams

from .samples import Samples, SamplesFormat


@dataclass
class TranscriptionBlock:
    init_prompt: str | None = None
    text: str | None = None
    init_data_b2: bytearray = field(default_factory=bytearray)
    samples: Samples | None = None


@dataclass
class TranscriptionState:
    pcm_params: PcmParams = field(default_factory=lambda: WHISPER_PRESET)
    blocks: list[TranscriptionBlock] | None = None
    processing: bool = False
    latency_ratio: float = 0.0
    text_finalized: list[str] = field(default_factory=list)

    def __post_init__(self):
        self.blocks = [TranscriptionBlock()]

    def append_chunk(
        self, chunk: bytes | bytearray, latency_ratio: float = 0.0
    ):
        assert not self.processing
        assert len(self.blocks) == 1
        block = self.blocks[0]
        block.text = None
        block.init_data_b2.extend(chunk)

        block.samples = Samples(
            block.init_data_b2, SamplesFormat.BINARY, self.pcm_params
        )
        self.latency_ratio = latency_ratio

    def __irshift__(self, trans: "TranscriptionStateTransformation"):
        trans.change_state(self)
        return self


class TranscriptionStateTransformation:
    def change_state(self, state: TranscriptionState):
        raise NotImplementedError

    def __rshift__(self, other: "TranscriptionStateTransformation"):
        return TranscriptionStateTransformationComposition(self, other)


class TranscriptionStateTransformationComposition(
    TranscriptionStateTransformation
):
    def __init__(
        self,
        t1: TranscriptionStateTransformation,
        t2: TranscriptionStateTransformation,
    ):
        self.t1 = t1
        self.t2 = t2

    def change_state(self, state: TranscriptionState):
        self.t1.change_state(state)
        self.t2.change_state(state)
