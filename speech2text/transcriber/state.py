from dataclasses import dataclass
from enum import Enum
from typing import List, Tuple

from speech2text.audio_data import NpData, PcmParams, PdData, WavData


class InvalidWorkflowStateException(Exception):
    pass


class Status(Enum):
    """FINALIZED -> INCREMENTED -> ADJUSTED -> SPLITTED -> REFINED -> FINALIZED"""

    FINALIZED = 0
    INCREMENTED = 1  #  INPUT_UPDATED
    ADJUSTED = 2  # getting PdData
    SPLITTED = 3  #
    REFINED = 4  # getting np.array
    INVALID = 999


@dataclass
class State:
    input_pcm_params: PcmParams
    status: Status = Status.FINALIZED
    ongoing_raw: WavData = WavData()
    ongoing_seg: PdData | None = None
    to_be_finalized: List[PdData] = []
    ongoing_np: NpData | None = None
    finalized_blocks: List[Tuple[WavData, str]]

    def __post_init__(self):
        self.ongoing_raw._pcm_params = self.input_pcm_params

    def validate(self, status: Status = None, raise_exception=True) -> bool:
        status = status or self.status
        try:
            if status in (Status.FINALIZED, Status.INCREMENTED):
                assert isinstance(self.ongoing_raw, WavData)
                assert self.ongoing_seg is None
                assert len(self.to_be_finalized) == 0
                assert self.ongoing_np is None
            elif status == Status.ADJUSTED:
                assert isinstance(self.ongoing_raw, WavData)
                assert isinstance(self.ongoing_seg, PcmParams)
                assert len(self.to_be_finalized) == 0
                assert self.ongoing_np is None
            elif status == Status.SPLITTED:
                assert isinstance(self.ongoing_raw, WavData)
                assert self.ongoing_seg is None or isinstance(
                    self.ongoing_seg, PcmParams
                )
                assert len(self.to_be_finalized) >= 0  # sic!
                assert self.ongoing_np is None
            elif status == Status.REFINED:
                assert isinstance(self.ongoing_raw, WavData)
                assert self.ongoing_seg is None or isinstance(
                    self.ongoing_seg, PcmParams
                )
                assert len(self.to_be_finalized) >= 0  # sic!
                assert self.ongoing_np is None or isinstance(
                    self.ongoing_np, NpData
                )
            else:
                raise InvalidWorkflowStateException
            return True
        except:
            if raise_exception:
                raise InvalidWorkflowStateException
            return False
