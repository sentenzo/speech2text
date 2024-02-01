"""Transcription is the last stage of the Speech-to-Text workflow.

Status: `TRANSCRIBED or SKIPPED -> FINALIZED`

Operations:
- cleanup
"""

from ..state import State, Status
from .stage import IStage


class AFinalizationStage(IStage):
    def _check_in_contract(self, state: State, *args, **kwargs):
        assert state.status in (Status.TRANSCRIBED, Status.SKIPPED)
        state.validate()

    def _check_out_contract(self, state: State, *args, **kwargs):
        assert state.status == Status.FINALIZED
        state.validate()


class FinalizationStage(AFinalizationStage):
    def _apply(self, state: State) -> State:
        state.latency_ratio = 0.0
        state.to_be_finalized = []
        state.status = Status.FINALIZED
