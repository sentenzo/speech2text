"""Incrementing is the 1st stage of the Speech-to-Text workflow.

Status: `FINALIZED -> INCREMENTED`

Operations:
- adding a new data-chunk to the raw WAVE data (incrementing it)
"""

from ..state import State, Status
from .stage import IStage


class AIncrementStage(IStage):
    def _check_in_contract(self, state: State, *args, **kwargs):
        assert state.status == Status.FINALIZED
        state.validate()

    def _check_out_contract(self, state: State, *args, **kwargs):
        assert state.status == Status.INCREMENTED
        state.validate()


class IncrementStage(AIncrementStage):
    def _apply(
        self,
        state: State,
        chunk: bytes | bytearray,
        latency_ratio: float = 0.0,
    ) -> State:
        state.ongoing.raw_data.append_chunk(chunk)
        state.latency_ratio = latency_ratio
        state.status = Status.INCREMENTED
        return state
