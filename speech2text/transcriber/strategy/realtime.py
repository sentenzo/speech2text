from pydub.generators import WhiteNoise
from speech2text.audio_data import NpData, PdData
from speech2text.audio_data.wave_data import WavData
from speech2text.settings import PyDubSettings, WhisperSettings, app_settings

from ..noisereduce import reduce_noise
from ..state import Block, State, Status
from ..whisper import transcribe
from .strategy import IStrategy


class RealtimeProcessing(IStrategy):
    def cold_start(self, state: State):
        assert len(state.ongoing.raw_data.raw_data) == 0
        duration_msec = 1000
        noise_chunk = (
            PdData(WhiteNoise().to_audio_segment(duration=duration_msec))
            .adjust_pcm_params(state.input_pcm_params)
            .raw_data
        )

        self.process_chunk(state, noise_chunk)
        state.ongoing = Block(WavData(state.input_pcm_params))

    def process_chunk(
        self,
        state: State,
        chunk: bytes | bytearray,
        latency_ratio: float = 0.0,
    ) -> State:
        """
        ```
        +> FINALIZED
        |  ↓  increment()
        |  INCREMENTED
        |  ↓  adjust()
        |  ADJUSTED
        |  ↓  split() ------+
        |  SPLITTED         |
        |  ↓  refine()      ↓
        |  REFINED       SKIPPED
        |  ↓  transcribe()  |
        |  TRANSCRIBED      |
        +---- finalize() ←--+
        ```
        """
        state.status = Status.FINALIZED  ###############################
        state.ongoing.raw_data.append_chunk(chunk)
        state.latency_ratio = latency_ratio
        state.status = Status.INCREMENTED  #############################
        state = self._adjust(state)
        state.status = Status.ADJUSTED  ################################
        state = self._split(state)
        if state.ongoing.seg_data is None:
            state.status = Status.SKIPPED  #############################
        else:
            state.status = Status.SPLITTED  ############################
            state = self._refine(state)
            state.status = Status.REFINED  #############################
            state = self._transcribe(state)
            state.status = Status.TRANSCRIBED  #########################
        state.status = Status.FINALIZED  ###############################

        return state

    def _adjust(self, state: State) -> State:
        settings = app_settings.transcriber.stages.adjust.pydub
        seg_data = PdData.load_from_wav_file(state.ongoing.raw_data)
        if settings.low_pass_filter:
            seg_data = seg_data.low_pass_filter(settings.low_pass_filter)
        if settings.high_pass_filter:
            seg_data = seg_data.high_pass_filter(settings.high_pass_filter)
        if settings.speed_up and settings.speed_up > 1.0:
            seg_data = seg_data.speedup(settings.speed_up)
        if settings.volume_up:
            seg_data += settings.volume_up
        if settings.normalize:
            seg_data = seg_data.normalize()
        state.ongoing.seg_data = seg_data
        return state

    def _split(self, state: State) -> State:
        settings = app_settings.transcriber.stages.split
        threshold = settings.agressive_threshold
        agressive = (
            state.latency_ratio > threshold.latency_ratio
            or state.ongoing.seg_data.duration_seconds > threshold.duration_sec
        )
        if agressive:
            for preset_name, preset_params in settings.pydub_split_on_silence:
                state = self._apply_split(state, preset_params)
                if (
                    state.ongoing.seg_data is None
                    or len(state.to_be_finalized) > 0
                ):
                    break
            else:
                state = self._apply_force_split(state)
        else:
            state = self._apply_split(
                state, settings.pydub_split_on_silence.default
            )
        return state

    def _apply_split(self, state: State, split_params) -> State:
        segments = state.ongoing.seg_data.split_on_silence(**split_params)
        if len(segments) == 0:  # only silence was found
            state.to_be_finalized = []
            state.ongoing.seg_data = None
            state.ongoing.raw_data = WavData(state.input_pcm_params)
        elif len(segments) == 1:  # no splitting, but maybe trimming
            state.to_be_finalized = []
            state.ongoing.seg_data = segments[0]
        elif len(segments) > 1:
            blocks = [
                Block.load_from_seg_data(segment) for segment in segments
            ]
            state.to_be_finalized = blocks[:-1]
            state.ongoing = blocks[-1]
        return state

    def _apply_force_split(self, state: State) -> State:
        l = len(state.ongoing.seg_data)

        # cut in half:
        left = state.ongoing.seg_data[: l // 2]
        right = state.ongoing.seg_data[l // 2 :]

        state.to_be_finalized = [
            Block.load_from_seg_data(left, state.input_pcm_params)
        ]
        state.ongoing.seg_data = right
        return state

    def _refine(self, state: State) -> State:
        settings_final = app_settings.transcriber.stages.refine.final

        def apply_pydub(block: Block, pydub_params: PyDubSettings):
            if pydub_params.low_pass_filter:
                block.seg_data = block.seg_data.low_pass_filter(
                    pydub_params.low_pass_filter
                )
            if pydub_params.high_pass_filter:
                block.seg_data = block.seg_data.high_pass_filter(
                    pydub_params.high_pass_filter
                )
            if pydub_params.volume_up:
                block.seg_data += pydub_params.volume_up
            if pydub_params.speed_up:
                block.seg_data = block.seg_data.speedup(pydub_params.speed_up)
            if pydub_params.normalize:
                block.seg_data = block.seg_data.normalize()

        # to_be_finalized
        if settings_final.pydub:
            for block in state.to_be_finalized:
                apply_pydub(block, settings_final.pydub)
        for block in state.to_be_finalized:
            block.arr_data = NpData.load_from_pd_data(block.seg_data)
        if settings_final.noisereduce:
            for block in state.to_be_finalized:
                block.arr_data = reduce_noise(block.arr_data)

        # ongoing
        settings_ongoing = app_settings.transcriber.stages.refine.ongoing
        if settings_ongoing.pydub:
            apply_pydub(state.ongoing, settings_ongoing.pydub)
        state.ongoing.arr_data = NpData.load_from_pd_data(
            state.ongoing.seg_data
        )
        if settings_ongoing.noisereduce:
            state.ongoing.arr_data = reduce_noise(state.ongoing.arr_data)

        return state

    def _transcribe(self, state: State) -> State:
        settings_final = app_settings.transcriber.stages.transcribe.final

        def apply_whisper(
            block: Block, whisper_params: WhisperSettings, **kwargs
        ):
            whisper_output = transcribe(
                block.arr_data,
                **whisper_params.model_dump().update(**kwargs),
            )
            block.text = whisper_output["text"]

        for block in state.to_be_finalized:
            apply_whisper(
                block, settings_final.whisper, condition_on_previous_text=True
            )
            state.finalized.append(block)

        settings_final = app_settings.transcriber.stages.transcribe.ongoing
        if len(state.finalized) > 0:
            state.ongoing_init_prompt = state.finalized[-1].text

        apply_whisper(
            state.ongoing,
            settings_final.whisper,
            condition_on_previous_text=False,
            initial_prompt=state.ongoing_init_prompt,
        )
