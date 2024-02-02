"""Adjustment is the 2d stage of the Speech-to-Text workflow.

Status: `INCREMENTED -> ADJUSTED`

Operations:
- converting WavData to PdData
- converting to Whisper-supported PCM parameters
- (optional) human voice frequency amplification
- (optional) speeding up
- (optional) volume up
"""

from speech2text.audio_data import PdData

from ..state import State, Status
from .stage import IStage


class AAdjustmentStage(IStage):
    def _check_in_contract(self, state: State, *args, **kwargs):
        assert state.status == Status.INCREMENTED
        state.validate()

    def _check_out_contract(self, state: State, *args, **kwargs):
        assert state.status == Status.ADJUSTED
        state.validate()


class AdjustmentStage(AAdjustmentStage):
    def _apply(self, state: State) -> State:
        state.ongoing.seg_data = (
            PdData.load_from_wav_file(state.ongoing.raw_data)
            # .low_pass_filter(300)
            # .high_pass_filter(3500)
            .adjust_pcm_params()
        )
        state.status = Status.ADJUSTED
        return state
