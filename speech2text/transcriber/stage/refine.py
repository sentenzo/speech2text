"""Refinement is the 4th stage of the Speech-to-Text workflow.

Status: `SPLITTED -> REFINED or SKIP`

Operations:
- if `ongoing_seg is None`: skip until the finalization stage
- for all `to_be_finalized` blocks:
    - (optionally) apply finalizing adjustments (AudioSegment transformations)
        - like `normalize` or volume-up or speeding-up
    - convert to `NpData`
    - (optionally) apply finalizing refinments
        - like `reduce_noise`
    - put into `to_be_finalized_arrs`
- for the `ongoing` block:
    - (optionaly) if having long duration, apply speeding-up
    - convert to `NpData`
    - (optionally) apply `reduce_noise`
    - put to `ongoing_np`
"""

from speech2text.audio_data import NpData, PdData

from ..noisereduce import reduce_noise
from ..state import State, Status
from .stage import IStage


class ARefinementStage(IStage):
    def _check_in_contract(self, state: State, *args, **kwargs):
        assert state.status == Status.SPLITTED
        state.validate()

    def _check_out_contract(self, state: State, *args, **kwargs):
        assert state.status == Status.REFINED
        state.validate()


class RefinementStage(ARefinementStage):
    def _apply(self, state: State) -> State:
        for block in state.to_be_finalized:
            block.arr_data = self._adjust_final(block.seg_data)
        state.ongoing.arr_data = self._adjust(state.ongoing.seg_data)
        state.status = Status.ADJUSTED
        return state

    def _adjust(self, segment: PdData) -> NpData:
        return NpData.load_from_pd_data(segment)

    def _adjust_final(self, segment: PdData) -> NpData:
        data = NpData.load_from_pd_data(segment.normalize())
        data = reduce_noise(data)
        return data
