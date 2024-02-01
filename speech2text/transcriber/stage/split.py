"""Splitting is the 3d stage of the Speech-to-Text workflow.

Status: `ADJUSTED -> SPLITTED`

Operations:
- apply split-on-silence procedure
- (optionally) if having big latency_ratio or the accumulated segment is 
  too long — apply a more aggressive splitting procedure (to prevent growing 
  delay in application response)

Possible results:
- one or more splits
    - the last segment becomes a new `ongoing_seg`
    - the others become `to_be_finalized_segs`
- no splits, but maybe trimming the following silence
    - `ongoing_seg` is updated
    - `to_be_finalized_segs` stays empty
- no splits, the output is empty
    - `ongoing_seg` is considered to be fully silent
"""

from typing import List

from speech2text.audio_data import PdData, WavData

from ..state import State, Status
from .stage import IStage


class ASplittingStage(IStage):
    def _check_in_contract(self, state: State, *args, **kwargs):
        assert state.status == Status.ADJUSTED
        state.validate()

    def _check_out_contract(self, state: State, *args, **kwargs):
        assert state.status == Status.SPLITTED
        state.validate()


class SplittingStage(ASplittingStage):
    # def __init__(self) -> None:
    #     super().__init__()

    def _apply(self, state: State) -> State:
        agressive = (
            state.latency_ratio > 1.0
            or state.ongoing_seg.duration_seconds > 10.0
        )
        segments: List[PdData] = []
        if agressive:
            segments = SplittingStage._split_agressive(state.ongoing_seg)
        else:
            segments = SplittingStage._split(state.ongoing_seg)
        if len(segments) == 0:  # only silence was found
            state.ongoing_seg = None
            state.ongoing_raw = WavData()
        elif len(segments) == 1:  # no splitting, but maybe trimming
            state.ongoing_seg = segments[0]
        elif len(segments) > 1:
            *state.to_be_finalized_segs, state.ongoing_seg = segments
        state.status = Status.SPLITTED

    @staticmethod
    def _split(segment: PdData) -> List[PdData]:
        return segment.split_on_silence(
            min_silence_len=1000,
            silence_thresh=-30,
            keep_silence=800,
            seek_step=10,
        )

    @staticmethod
    def _split_agressive(segment: PdData) -> List[PdData]:
        return segment.split_on_silence(
            min_silence_len=500,
            silence_thresh=-28,
            keep_silence=500,
            seek_step=10,
        )
