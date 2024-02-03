from dataclasses import dataclass, field
from enum import Enum
from typing import List

from speech2text.audio_data import NpData, PcmParams, PdData, WavData


class InvalidWorkflowStateException(Exception):
    pass


class Status(Enum):
    """
    FINALIZED
    ↓
    INCREMENTED
    ↓
    ADJUSTED --------+
    ↓                |
    SPLITTED         |
    ↓                ↓
    REFINED       SKIPPED
    ↓                |
    TRANSCRIBED      |
    ↓                |
    FINALIZED ←------+
    ↓
    ...
    """

    FINALIZED = 0
    INCREMENTED = 1  #  INPUT_UPDATED
    ADJUSTED = 2  # getting PdData
    SPLITTED = 3  #
    REFINED = 4  # getting np.array
    TRANSCRIBED = 5
    SKIPPED = 100
    INVALID = 999


@dataclass
class Block:
    raw_data: WavData | None = None
    seg_data: PdData | None = None
    arr_data: NpData | None = None
    text: str | None = None

    def _has_raw(self):
        return isinstance(self.raw_data, WavData)

    def _has_seg(self):
        return isinstance(self.seg_data, PdData)

    def _has_arr(self):
        return isinstance(self.arr_data, NpData)

    def _has_text(self):
        return isinstance(self.text, str)


@dataclass
class State:
    input_pcm_params: PcmParams
    latency_ratio: float = 0.0
    _status: Status = Status.FINALIZED
    ongoing: Block = field(default_factory=lambda: Block(WavData()))
    ongoing_init_prompt: str | None = None
    to_be_finalized: List[Block] = field(default_factory=list)
    finalized: List[Block] = field(default_factory=list)

    def __post_init__(self):
        self.ongoing.raw_data._pcm_params = self.input_pcm_params

    def _validate_finalized(self) -> None:
        assert self.ongoing._has_raw()
        assert self.to_be_finalized == []

    def _validate_incremented(self) -> None:
        pass

    def _validate_adjusted(self) -> None:
        assert self.ongoing._has_seg()

    def _validate_splitted(self) -> None:
        assert self.ongoing._has_seg()

    def _validate_refined(self) -> None:
        assert self.ongoing._has_seg()
        assert self.ongoing._has_arr()

    def _validate_transcribed(self) -> None:
        assert self.ongoing._has_text()

    def _validate_skipped(self) -> None:
        pass

    def _validate_invalid(self) -> None:
        raise InvalidWorkflowStateException

    def validate(self, raise_exception=True) -> bool:
        try:
            {
                Status.FINALIZED: self._validate_finalized,
                Status.INCREMENTED: self._validate_incremented,
                Status.ADJUSTED: self._validate_adjusted,
                Status.SPLITTED: self._validate_splitted,
                Status.REFINED: self._validate_refined,
                Status.TRANSCRIBED: self._validate_transcribed,
                Status.SKIPPED: self._validate_skipped,
                Status.INVALID: self._validate_invalid,
            }[self._status]()
            return True
        except:
            if raise_exception:
                raise InvalidWorkflowStateException
            return False

    @property
    def status(self) -> Status:
        return self._status

    @status.setter
    def status(self, new_status: Status):
        old_status = self._status
        try:
            self._status = new_status
            self.validate()
        except:
            self._status = old_status
            raise
