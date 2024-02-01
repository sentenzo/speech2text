"""Transcription is the 5th and the main stage of the Speech-to-Text workflow.

Status: `REFINED -> TRANSCRIBED`

Operations:
- for all `to_be_finalized` blocks:
    - apply Whisper transcription with final preset
- for `ongoing` block:
    - apply Whisper transcription with ongoing preset
"""

from speech2text.audio_data import NpData

from ..state import State, Status
from ..whisper import DEFAULT_TRANSCRIPTION_PARAMETERS, ModelName, transcribe
from .stage import IStage


class ATranscriptionStage(IStage):
    def _check_in_contract(self, state: State, *args, **kwargs):
        assert state.status == Status.REFINED
        state.validate()

    def _check_out_contract(self, state: State, *args, **kwargs):
        assert state.status == Status.TRANSCRIBED
        state.validate()


class TranscriptionStage(ATranscriptionStage):
    ongoing_whisper_params = DEFAULT_TRANSCRIPTION_PARAMETERS.replace(
        condition_on_previous_text=False
    )
    finalize_whisper_params = DEFAULT_TRANSCRIPTION_PARAMETERS

    def _apply(self, state: State) -> State:
        for block in state.to_be_finalized:
            block.text = self._transcribe_final(block.arr_data)
            state.finalized.append(block)

        if state.finalized:
            state.ongoing_init_prompt = state.finalized[-1].text
        state.ongoing.text = self._transcribe(
            state.ongoing.arr_data,
            state.ongoing_init_prompt,
        )
        state.status = Status.TRANSCRIBED
        return state

    def _transcribe(self, array: NpData, init_prompt=None) -> str:
        whisper_output = transcribe(
            array,
            ModelName.TINY_EN,
            self.ongoing_whisper_params.replace(initial_prompt=init_prompt),
        )
        return whisper_output["text"]

    def _transcribe_final(self, array: NpData) -> str:
        whisper_output = transcribe(
            array,
            ModelName.SMALL_EN,
            self.finalize_whisper_params,
        )
        return whisper_output["text"]
